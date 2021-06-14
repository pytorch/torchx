#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Trainer App Example
=============================================

This is an example TorchX app that uses PyTorch Lightning and ClassyVision to
train a model.

This app only uses standard OSS libraries and has no runtime torchx
dependencies. For saving and loading data and models it uses fsspec which makes
the app agnostic to the environment it's running in.
"""

import argparse
import sys
import tempfile
from typing import List

import pytorch_lightning as pl
from data import TinyImageNetDataModule, download_data
from model import TinyImageNetModel, export_inference_model
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger


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


def main(argv: List[str]) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        args = parse_args(argv)

        # Init our model
        model = TinyImageNetModel()

        # Download and setup the data module
        data_path = download_data(args.data_path, tmpdir)
        data = TinyImageNetDataModule(
            data_dir=data_path,
            batch_size=args.batch_size,
        )

        # Setup model checkpointing
        checkpoint_callback = ModelCheckpoint(
            monitor="train_loss",
            dirpath=args.output_path,
            save_last=True,
        )
        if args.load_path:
            print(f"loading checkpoint: {args.load_path}...")
            model.load_from_checkpoint(checkpoint_path=args.load_path)

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
        trainer.fit(model, data)

        # Export the inference model
        export_inference_model(model, args.output_path, tmpdir)


if __name__ == "__main__":
    main(sys.argv[1:])
