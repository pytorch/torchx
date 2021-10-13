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

from typing import Optional, Dict, List

import torchx.specs.api as torchx
from torchx.specs import named_resources


def trainer(
    image: str,
    output_path: str,
    data_path: Optional[str] = None,
    load_path: str = "",
    log_path: str = "/logs",
    resource: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    skip_export: bool = False,
    epochs: int = 1,
    layers: Optional[List[int]] = None,
    learning_rate: Optional[float] = None,
    num_samples: int = 200,
) -> torchx.AppDef:
    """Runs the example lightning_classy_vision app.

    Args:
        image: image to run (e.g. foobar:latest)
        output_path: output path for model checkpoints (e.g. file:///foo/bar)
        load_path: path to load pretrained model from
        data_path: path to the data to load, if data_path is not provided,
            auto generated test data will be used
        log_path: path to save tensorboard logs to
        resource: the resources to use
        env: env variables for the app
        skip_export: disable model export
        epochs: number of epochs to run
        layers: the number of convolutional layers and their number of channels
        learning_rate: the learning rate to use for training
        num_samples: the number of images to run per batch, use 0 to run on all
    """
    env = env or {}
    args = [
        "-m",
        "torchx.examples.apps.lightning_classy_vision.train",
        "--output_path",
        output_path,
        "--load_path",
        load_path,
        "--log_path",
        log_path,
        "--epochs",
        str(epochs),
    ]
    if layers:
        args += ["--layers"] + [str(layer) for layer in layers]
    if learning_rate:
        args += ["--lr", str(learning_rate)]
    if num_samples and num_samples > 0:
        args += ["--num_samples", str(num_samples)]
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
                entrypoint="python",
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
) -> torchx.AppDef:
    """Runs the model interpretability app on the model outputted by the training
    component.

    Args:
        image: image to run (e.g. foobar:latest)
        load_path: path to load pretrained model from
        data_path: path to the data to load
        output_path: output path for model checkpoints (e.g. file:///foo/bar)
        resource: the resources to use
    """
    return torchx.AppDef(
        name="cv-interpret",
        roles=[
            torchx.Role(
                name="worker",
                entrypoint="python",
                args=[
                    "-m",
                    "torchx.examples.apps.lightning_classy_vision.interpret",
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
