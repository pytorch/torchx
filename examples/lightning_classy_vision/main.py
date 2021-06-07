#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-unsafe

import argparse
import os.path
import subprocess
import sys
import tarfile
import tempfile
from typing import List

import fsspec
import pytorch_lightning as pl
import torch
import torch.jit
from classy_vision.dataset.classy_dataset import ClassyDataset
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class TinyImageNetDataset(ClassyDataset):
    def __init__(self, data_path, transform):
        batchsize_per_replica = 16
        shuffle = False
        num_samples = 1000
        dataset = datasets.ImageFolder(data_path)
        super().__init__(
            dataset, batchsize_per_replica, shuffle, transform, num_samples
        )


class TinyImageNetModel(pl.LightningModule):
    """
    An very simple linear model for the tiny image net dataset.
    """

    def __init__(self):
        super().__init__()
        self.l1 = torch.nn.Linear(64 * 64, 4096)

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
        "--batch_size", type=int, default=32, help="batch size to use for training"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="path to load the training data from",
        required=True,
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


def download_data(remote_path: str, tmpdir: str) -> str:
    tar_path = os.path.join(tmpdir, "data.tar.gz")
    print(f"downloading dataset from {remote_path} to {tar_path}...")
    fs, _, rpaths = fsspec.get_fs_token_paths(remote_path)
    assert len(rpaths) == 1, "must have single path"
    fs.get(rpaths[0], tar_path)

    data_path = os.path.join(tmpdir, "data")
    print(f"extracting {tar_path} to {data_path}...")
    with tarfile.open(tar_path, mode="r") as f:
        f.extractall(data_path)

    return data_path


def export_inference_model(
    model: TinyImageNetModel, out_path: str, tmpdir: str
) -> None:
    print("exporting inference model")
    jit_path = os.path.join(tmpdir, "model_jit.pt")
    jitted = torch.jit.script(model)
    print(f"saving JIT model to {jit_path}")
    torch.jit.save(jitted, jit_path)

    model_name = "tiny_image_net"

    mar_path = os.path.join(tmpdir, f"{model_name}.mar")
    print(f"creating model archive at {mar_path}")
    subprocess.run(
        [
            "torch-model-archiver",
            "--model-name",
            "tiny_image_net",
            "--handler",
            "lightning_classy_vision/handler.py",
            "--version",
            "1",
            "--serialized-file",
            jit_path,
            "--export-path",
            tmpdir,
        ],
        check=True,
    )

    remote_path = os.path.join(out_path, "model.mar")
    print(f"uploading to {remote_path}")
    fs, _, rpaths = fsspec.get_fs_token_paths(remote_path)
    assert len(rpaths) == 1, "must have single path"
    fs.put(mar_path, rpaths[0])


def main(argv):
    with tempfile.TemporaryDirectory() as tmpdir:
        args = parse_args(argv)

        # Init our model
        model = TinyImageNetModel()

        data_path = download_data(args.data_path, tmpdir)

        # Setup data loader and transforms
        img_transform = transforms.Compose(
            [
                transforms.Grayscale(),
                transforms.ToTensor(),
            ]
        )
        train_ds = TinyImageNetDataset(
            data_path=os.path.join(data_path, "train"),
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

        logger = TensorBoardLogger(
            save_dir=args.log_dir, version=1, name="lightning_logs"
        )

        # Initialize a trainer
        trainer = pl.Trainer(
            logger=logger,
            max_epochs=args.epochs,
            callbacks=[checkpoint_callback],
        )

        # Train the model ⚡
        trainer.fit(model, train_loader)

        export_inference_model(model, args.output_path, tmpdir)


if __name__ == "__main__":
    main(sys.argv[1:])
