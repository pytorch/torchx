#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import argparse
import sys
from typing import List

import pytorch_lightning as pl
import torch
from classy_vision.dataset.classy_dataset import ClassyDataset
from classy_vision.dataset.core.random_image_datasets import (
    RandomImageDataset,
    SampleType,
)
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms


class SyntheticMNIST(ClassyDataset):
    def __init__(self, transform):
        batchsize_per_replica = 16
        shuffle = True
        num_samples = 1000
        dataset = RandomImageDataset(
            crop_size=28,
            num_channels=3,
            num_samples=num_samples,
            num_classes=10,
            seed=1234,
            sample_type=SampleType.TUPLE,
        )
        super().__init__(
            dataset, batchsize_per_replica, shuffle, transform, num_samples
        )


class MNISTModel(pl.LightningModule):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_nb):
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.02)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="pytorch lightning + classy vision TorchX example app"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="number of epochs to train"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size to use for traiing"
    )
    parser.add_argument("--load_path", type=str, help="checkpoint path to load from")
    parser.add_argument(
        "--output_path",
        type=str,
        help="path to place checkpoints and model outputs",
        required=True,
    )
    parser.add_argument(
        "--log_dir", type=str, help="directory to place the logs", default="/tmp"
    )

    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv)

    # Init our model
    mnist_model = MNISTModel()

    # Init DataLoader from MNIST Dataset
    img_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.ToTensor(),
        ]
    )
    train_ds = SyntheticMNIST(
        transform=lambda x: (img_transform(x[0]), x[1]),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size)

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        dirpath=args.output_path,
        save_last=True,
    )
    if args.load_path:
        print(f"loading checkpoint: {args.load_path}...")
        mnist_model.load_from_checkpoint(checkpoint_path=args.load_path)

    logger = TensorBoardLogger(save_dir=args.log_dir, version=1, name="lightning_logs")

    # Initialize a trainer
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback],
    )

    # Train the model ⚡
    trainer.fit(mnist_model, train_loader)


if __name__ == "__main__":
    main(sys.argv[1:])
