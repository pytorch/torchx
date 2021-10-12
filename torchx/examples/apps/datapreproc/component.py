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


def data_preproc(
    image: str,
    output_path: str,
    input_path: str = "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    env: Optional[Dict[str, str]] = None,
    resource: Optional[str] = None,
) -> specs.AppDef:
    """Data PreProc app.

    Data PreProc app.

    Args:
        image: Image to use
        output_path: Url-like path to save the processes compressed images
        input_path: Url-like path to fetch the imagenet dataset
        env: Env variables to transfer to the user script
        resource: String representation of the resource
        dryrun: Starts the app, but does not actually perform any work.

    Returns:
        specs.AppDef: Torchx AppDef
    """

    env = env or {}
    args = [
        "-m",
        "torchx.examples.apps.datapreproc.datapreproc",
        "--input_path",
        input_path,
        "--output_path",
        output_path,
    ]

    if resource:
        resource_def = specs.named_resources[resource]
    else:
        resource_def = specs.Resource(cpu=1, gpu=0, memMB=1024)
    return specs.AppDef(
        name="datapreproc",
        roles=[
            specs.Role(
                name="worker",
                image=image,
                entrypoint="python",
                args=args,
                env=env,
                resource=resource_def,
            )
        ],
    )
