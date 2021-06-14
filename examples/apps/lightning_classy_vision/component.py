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

from typing import Optional

import torchx.specs.api as torchx
from torchx.components.base import named_resource
from torchx.components.base.binary_component import binary_component


def trainer(
    image: str,
    output_path: str,
    data_path: str,
    load_path: str = "",
    log_dir: str = "/logs",
    resource: Optional[str] = None,
) -> torchx.AppDef:
    """Runs the example lightning_classy_vision app.

    Args:
        image: image to run (e.g. foobar:latest)
        output_path: output path for model checkpoints (e.g. file:///foo/bar)
        load_path: path to load pretrained model from
        data_path: path to the data to load
        log_dir: path to save tensorboard logs to
        resource: the resources to use
    """
    return binary_component(
        name="examples-lightning_classy_vision-trainer",
        entrypoint="lightning_classy_vision/train.py",
        args=[
            "--output_path",
            output_path,
            "--load_path",
            load_path,
            "--log_dir",
            log_dir,
            "--data_path",
            data_path,
        ],
        image=image,
        resource=named_resource(resource)
        if resource
        else torchx.Resource(cpu=1, gpu=0, memMB=1024),
    )


def interpret(
    image: str,
    load_path: str,
    data_path: str,
    output_path: str,
    resource: Optional[str] = None,
) -> torchx.AppDef:
    """Runs the model intepretability app on the model outputted by the training
    component.

    Args:
        image: image to run (e.g. foobar:latest)
        load_path: path to load pretrained model from
        data_path: path to the data to load
        output_path: output path for model checkpoints (e.g. file:///foo/bar)
        resource: the resources to use
    """
    return binary_component(
        name="examples-lightning_classy_vision-intepret",
        entrypoint="lightning_classy_vision/interpret.py",
        args=[
            "--load_path",
            load_path,
            "--data_path",
            data_path,
            "--output_path",
            output_path,
        ],
        image=image,
        resource=named_resource(resource)
        if resource
        else torchx.Resource(cpu=1, gpu=0, memMB=1024),
    )
