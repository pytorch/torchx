#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Distributed Trainer Example
=============================================

The example of the DDP train job that user `torch.distributed.run` to
start processes. `torch.distributed.run` exposes env. variables that
are used by this script.

Example of running this job on the kubernetes scheduler:

.. code-block:: shell-session
    $ torchx run --scheduler kubernetes \
    --scheduler_args namespace=default,queue=test \
    dist.ddp --image $YOUR_IMAGE \
    --entrypoint dist_cifar10/trainer.py \
    --rdzv_backend=etcd-v2 --rdzv_endpoint=etcd-server:2379 \
    --nnodes 2 \
    -- --epochs 1

"""

import argparse
import os
import sys
import tempfile
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from torchvision.datasets import CIFAR10

NUM_WORKERS: int = max(1, (os.cpu_count() or 1) // 2)

# pyre-ignore-all-errors[16]


def create_model(num_classes: int = 10) -> nn.Module:
    model = torchvision.models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    nfeatures = model.fc.in_features
    model.fc = nn.Linear(nfeatures, num_classes)
    return model


def create_train_valid_data_loaders(
    dest_dir: str, batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    normalize = torchvision.transforms.Normalize(
        mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    )
    train_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            normalize,
        ]
    )

    valid_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            normalize,
        ]
    )

    train_dataset = CIFAR10(
        root=dest_dir, train=True, download=True, transform=train_transforms
    )
    valid_dataset = CIFAR10(
        root=dest_dir, train=False, download=True, transform=valid_transforms
    )

    train_dl = DataLoader(train_dataset, batch_size=batch_size, num_workers=NUM_WORKERS)
    valid_dl = DataLoader(valid_dataset, batch_size=batch_size, num_workers=NUM_WORKERS)
    return train_dl, valid_dl


class LitResnet(LightningModule):
    def __init__(self, lr: float = 0.05) -> None:
        super().__init__()

        self.save_hyperparameters()
        self.model: nn.Module = create_model()

    # pyre-ignore[14]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return F.log_softmax(out, dim=1)

    # pyre-ignore[14]
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(
        self, batch: Tuple[torch.Tensor, torch.Tensor], stage: Optional[str] = None
    ) -> None:
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)

    # pyre-ignore[14]
    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        self.evaluate(batch, "val")

    # pyre-ignore[14]
    def test_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        self.evaluate(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        return optimizer


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Trainer that trains the last layer of pretrained resnet18 for cifar10 classification"
    )
    parser.add_argument(
        "--dryrun",
        help=argparse.SUPPRESS,
        action="store_true",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size to use for training"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="path to place checkpoints",
        required=True,
    )

    return parser.parse_args(argv)


def get_gpu_devices() -> int:
    return torch.cuda.device_count()


def main() -> None:
    args = parse_args(sys.argv[1:])
    if args.dryrun:
        print("App dist_cifar.train started successfully")
        return
    gpus = get_gpu_devices()
    batch_size = args.batch_size
    num_nodes = int(os.environ["GROUP_WORLD_SIZE"])
    with tempfile.TemporaryDirectory() as tmpdir:
        train_dl, valid_dl = create_train_valid_data_loaders(tmpdir, batch_size)
        model = LitResnet(lr=0.05)

        checkpoint_callback = ModelCheckpoint(
            monitor="train_loss",
            dirpath=args.output_path,
            save_last=True,
        )

        Trainer(
            num_nodes=num_nodes,
            gpus=gpus,
            accelerator="ddp",
            max_epochs=args.epochs,
            callbacks=[checkpoint_callback],
        ).fit(model, train_dataloaders=train_dl, val_dataloaders=valid_dl)


if __name__ == "__main__":
    main()
