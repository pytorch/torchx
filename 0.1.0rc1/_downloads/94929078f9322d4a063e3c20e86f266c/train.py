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
import os
import sys
import tempfile
from typing import List, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# ensure data and module are on the path
sys.path.append(".")

from torchx.examples.apps.lightning_classy_vision.data import (
    TinyImageNetDataModule,
    download_data,
    create_random_data,
)
from torchx.examples.apps.lightning_classy_vision.model import (
    TinyImageNetModel,
    export_inference_model,
)
from torchx.examples.apps.lightning_classy_vision.profiler import (
    SimpleLoggingProfiler,
)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="pytorch lightning + classy vision TorchX example app"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="number of epochs to train"
    )
    parser.add_argument("--lr", type=float, help="learning rate")
    parser.add_argument(
        "--batch_size", type=int, default=32, help="batch size to use for training"
    )
    parser.add_argument("--num_samples", type=int, default=None, help="num_samples")
    parser.add_argument(
        "--test",
        help="Sets to test mode, training on a much smaller set of randomly generated images",
        action="store_true",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="path to load the training data from, if not provided, random data will be generated",
    )
    parser.add_argument("--skip_export", action="store_true")
    parser.add_argument("--load_path", type=str, help="checkpoint path to load from")
    parser.add_argument(
        "--output_path",
        type=str,
        help="path to place checkpoints and model outputs, if not specified, checkpoints are not saved",
    )
    parser.add_argument(
        "--log_path",
        type=str,
        help="path to place the tensorboard logs",
        default="/tmp",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        help="the MLP hidden layers and sizes, used for neural architecture search",
    )
    return parser.parse_args(argv)


def get_gpu_devices() -> int:
    return torch.cuda.device_count()


def get_model_checkpoint(args: argparse.Namespace) -> Optional[ModelCheckpoint]:
    if not args.output_path:
        return None
    return ModelCheckpoint(
        monitor="train_loss",
        dirpath=args.output_path,
        save_last=True,
    )


def main(argv: List[str]) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        args = parse_args(argv)

        # Init our model
        model = TinyImageNetModel(args.layers)
        print(model)

        # Download and setup the data module
        if not args.data_path:
            data_path = os.path.join(tmpdir, "data")
            os.makedirs(data_path)
            create_random_data(data_path)
        else:
            data_path = download_data(args.data_path, tmpdir)

        data = TinyImageNetDataModule(
            data_dir=data_path,
            batch_size=args.batch_size,
            num_samples=5 if args.test else args.num_samples,
        )

        # Setup model checkpointing
        checkpoint_callback = get_model_checkpoint(args)
        callbacks = []
        if checkpoint_callback:
            callbacks.append(checkpoint_callback)
        if args.load_path:
            print(f"loading checkpoint: {args.load_path}...")
            model.load_from_checkpoint(checkpoint_path=args.load_path)

        logger = TensorBoardLogger(
            save_dir=args.log_path, version=1, name="lightning_logs"
        )

        # Initialize a trainer
        num_nodes = int(os.environ.get("GROUP_WORLD_SIZE", 1))
        trainer = pl.Trainer(
            num_nodes=num_nodes,
            accelerator="ddp2",
            logger=logger,
            max_epochs=args.epochs,
            callbacks=[checkpoint_callback],
            profiler=SimpleLoggingProfiler(logger),
        )

        # Train the model ⚡
        trainer.fit(model, data)
        print(
            f"train acc: {model.train_acc.compute()}, val acc: {model.val_acc.compute()}"
        )

        if not args.skip_export:
            # Export the inference model
            export_inference_model(model, args.output_path, tmpdir)


if __name__ == "__main__" and "NOTEBOOK" not in globals():
    main(sys.argv[1:])
