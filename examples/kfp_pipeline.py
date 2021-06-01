#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
from typing import List

import kfp
from torchx.components.base.binary_component import binary_component
from torchx.pipelines.kfp.adapter import ContainerFactory, component_from_app


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="example kfp pipeline")
    parser.add_argument(
        "--image",
        type=str,
        help="docker image to use",
        default="495572122715.dkr.ecr.us-west-2.amazonaws.com/torchx/examples:latest",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        help="path to place the data",
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
    args = parse_args(argv)

    datapreproc_app = binary_component(
        name="examples-datapreproc",
        entrypoint="datapreproc/main.py",
        args=[
            "--output_path",
            args.data_path,
        ],
        image=args.image,
    )
    datapreproc_comp: ContainerFactory = component_from_app(datapreproc_app)

    trainer_app = binary_component(
        name="examples-lightning_classy_vision-trainer",
        entrypoint="lightning_classy_vision/main.py",
        args=[
            "--output_path",
            args.output_path,
            "--load_path",
            args.load_path or "",
            "--log_dir",
            args.log_dir,
            "--data_path",
            args.data_path,
        ],
        image=args.image,
    )
    trainer_comp: ContainerFactory = component_from_app(trainer_app)

    def pipeline() -> None:
        datapreproc = datapreproc_comp()
        trainer = trainer_comp()
        trainer.after(datapreproc)

    kfp.compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="pipeline.yaml",
    )


if __name__ == "__main__":
    main(sys.argv[1:])
