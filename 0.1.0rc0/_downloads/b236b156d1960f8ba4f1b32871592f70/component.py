#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Datapreproc Component Example
=============================

This is a component definition that runs the example datapreproc app.
"""

from typing import Dict, Optional

import torchx.specs as specs
from torchx.components.base.binary_component import binary_component


def data_preproc(
    image: str,
    entrypoint: str,
    output_path: str,
    input_path: str = "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    input_md5: str = "90528d7ca1a48142e341f4ef8d21d0de",
    env: Optional[Dict[str, str]] = None,
    resource: Optional[str] = None,
    name: str = "datapreproc",
) -> specs.AppDef:
    """Data PreProc app.

    Data PreProc app.

    Args:
        image: Image to use
        entrypoint: User script to launch
        output_path: Url-like path to save the processes compressed images
        input_path: Url-like path to fetch the imagenet dataset
        input_md5: Hash of the input imagenet dataset file
        env: Env variables to transfer to the user script
        resource: String representation of the resource
        name: Name of the worker

    Returns:
        specs.AppDef: Torchx AppDef
    """

    env = env or {}
    args = [
        "--input_path",
        input_path,
        "--input_md5",
        input_md5,
        "--output_path",
        output_path,
    ]
    if resource:
        resource_def = specs.named_resources[resource]
    else:
        resource_def = specs.Resource(cpu=1, gpu=0, memMB=1024)

    return binary_component(
        name="datapreproc_role",
        entrypoint=entrypoint,
        args=args,
        env=env,
        image=image,
        resource=resource_def,
    )
