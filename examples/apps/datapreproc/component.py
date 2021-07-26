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
from torchx.components.base.torch_dist_component import torch_dist_component


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
        num_replicas=1,
    )

    return specs.AppDef(name, roles=[ddp_role])


def dist_data_preproc(
    image: str,
    entrypoint: str,
    output_path: str,
    input_path: str = "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    input_md5: str = "90528d7ca1a48142e341f4ef8d21d0de",
    base_image: Optional[str] = None,
    env: Optional[Dict[str, str]] = None,
    resource: str = "T1",
    nnodes: int = 1,
    nproc_per_node: int = 1,
) -> specs.AppDef:
    """Example of the custom datapreproc component that is used by the ``torchx.examples.workflow``
    with additional functionality:

        * The component requires pytorch library and user script is launched via ``torch.distirbuted.run``
        * The component provides ability to specify ``base_image``

    Args:
        image: image to run(e.g. foobar:latest)
        entrypoint: main script to run
        output_path: path to the dir that will be used to store data in format: manifold://${bucket}/foo/bar
        input_path: url that has imagenet dataset
        input_md5: hash of the dataset
        base_image: Optional base image, e.g. (pytorch:latest)
        env: Env variables to pass to the workers
        resource: ``torchx.specs.fb.named_resource`` string definitions
        nnodes_ number of nodes
        nproc_per_node: number of processes per node
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

    return torch_dist_component(
        name="datapreproc",
        image=image,
        entrypoint=entrypoint,
        nnodes=nnodes,
        nproc_per_node=nproc_per_node,
        base_image=base_image,
        args=args,
        env=env,
        resource=resource,
    )
