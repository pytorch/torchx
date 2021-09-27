# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Trainer Component Example
=========================

This is a component definition that runs the example lightning_classy_vision app.
"""

from typing import Optional, Dict

import torchx.specs.api as torchx
from torchx.specs import named_resources


def trainer(
    image: str,
    output_path: str,
    data_path: Optional[str] = None,
    entrypoint: str = "examples/apps/lightning_classy_vision/train.py",
    load_path: str = "",
    log_path: str = "/logs",
    resource: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    skip_export: bool = False,
    epochs: int = 1,
) -> torchx.AppDef:
    """Runs the example lightning_classy_vision app.

    Args:
        image: image to run (e.g. foobar:latest)
        output_path: output path for model checkpoints (e.g. file:///foo/bar)
        load_path: path to load pretrained model from
        data_path: path to the data to load, if data_path is not provided,
            auto generated test data will be used
        entrypoint: user script to launch.
        log_path: path to save tensorboard logs to
        resource: the resources to use
        env: env variables for the app
        skip_export: disable model export
        epochs: number of epochs to run
    """
    env = env or {}
    args = [
        "--output_path",
        output_path,
        "--load_path",
        load_path,
        "--log_path",
        log_path,
        "--epochs",
        str(epochs),
    ]
    if data_path:
        args += ["--data_path", data_path]
    else:
        args.append("--test")
    if skip_export:
        args.append("--skip_export")
    return torchx.AppDef(
        name="cv-trainer",
        roles=[
            torchx.Role(
                name="worker",
                entrypoint=entrypoint,
                args=args,
                env=env,
                image=image,
                resource=named_resources[resource]
                if resource
                else torchx.Resource(cpu=1, gpu=0, memMB=1500),
            )
        ],
    )


def interpret(
    image: str,
    load_path: str,
    data_path: str,
    output_path: str,
    resource: Optional[str] = None,
    entrypoint: str = "examples/apps/lightning_classy_vision/interpret.py",
) -> torchx.AppDef:
    """Runs the model interpretability app on the model outputted by the training
    component.

    Args:
        image: image to run (e.g. foobar:latest)
        load_path: path to load pretrained model from
        data_path: path to the data to load
        output_path: output path for model checkpoints (e.g. file:///foo/bar)
        resource: the resources to use
        entrypoint: user script to launch.
    """
    return torchx.AppDef(
        name="cv-interpret",
        roles=[
            torchx.Role(
                name="worker",
                entrypoint=entrypoint,
                args=[
                    "--load_path",
                    load_path,
                    "--data_path",
                    data_path,
                    "--output_path",
                    output_path,
                ],
                image=image,
                resource=named_resources[resource]
                if resource
                else torchx.Resource(cpu=1, gpu=0, memMB=1024),
            )
        ],
    )
