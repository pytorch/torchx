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
from torchx.components.base.binary_component import binary_component
from torchx.specs import named_resources


def trainer(
    image: str,
    output_path: str,
    data_path: str,
    entrypoint: str = "lightning_classy_vision/train.py",
    load_path: str = "",
    log_path: str = "/logs",
    resource: Optional[str] = None,
    nnodes: int = 1,
    env: Optional[Dict[str, str]] = None,
    nproc_per_node: int = 1,
    skip_export: bool = False,
) -> torchx.AppDef:
    """Runs the example lightning_classy_vision app.

    Args:
        image: image to run (e.g. foobar:latest)
        output_path: output path for model checkpoints (e.g. file:///foo/bar)
        load_path: path to load pretrained model from
        data_path: path to the data to load
        log_path: path to save tensorboard logs to
        resource: the resources to use
        nnodes: number of nodes
        env: env variables for the app
        nproc_per_node: number of processes per node
        skip_export: disable model export
    """
    env = env or {}
    args = [
        "--output_path",
        output_path,
        "--load_path",
        load_path,
        "--log_pat",
        log_path,
        "--data_path",
        data_path,
    ]
    if skip_export:
        args.append("--skip_export")
    return binary_component(
        name="examples-lightning_classy_vision-trainer",
        entrypoint=entrypoint,
        args=args,
        env=env,
        image=image,
        resource=named_resources[resource]
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
    """Runs the model interpretability app on the model outputted by the training
    component.

    Args:
        image: image to run (e.g. foobar:latest)
        load_path: path to load pretrained model from
        data_path: path to the data to load
        output_path: output path for model checkpoints (e.g. file:///foo/bar)
        resource: the resources to use
    """
    return binary_component(
        name="examples-lightning_classy_vision-interpret",
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
        resource=named_resources[resource]
        if resource
        else torchx.Resource(cpu=1, gpu=0, memMB=1024),
    )
