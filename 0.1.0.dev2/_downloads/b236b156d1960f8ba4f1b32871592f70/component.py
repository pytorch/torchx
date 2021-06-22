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
    entrypoint: str,
    output_path: str,
    input_path: str = "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    input_md5: str = "90528d7ca1a48142e341f4ef8d21d0de",
    env: Optional[Dict[str, str]] = None,
    name: str = "datapreproc",
) -> specs.AppDef:
    """Data PreProc app.

    Data PreProc app.

    Args:
        name: Name of the app.
        output_path: output_path

    Returns:
        specs.AppDef: Torchx AppDef
    """

    env = env or {}
    resource = specs.named_resources["T1"]
    ddp_role = specs.Role(
        name="datapreproc_role",
        image=image,
        entrypoint=entrypoint,
        args=[
            "--input_path",
            input_path,
            "--input_md5",
            input_md5,
            "--output_path",
            output_path,
        ],
        env=env,
        resource=resource,
    ).replicas(1)

    return specs.AppDef(name).of(ddp_role)
