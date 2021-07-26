# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Components for applications that run as distributed jobs. Many of the
components in this section are simply topological, meaning that they define
the layout of the nodes in a distributed setting and take the actual
binaries that each group of nodes (``specs.Role``) runs.
"""

from typing import Dict, Optional

import torchx.specs as specs
from torchx.components.base.torch_dist_component import torch_dist_component


def ddp(
    image: str,
    entrypoint: str,
    resource: Optional[str] = None,
    nnodes: int = 1,
    nproc_per_node: int = 1,
    base_image: Optional[str] = None,
    name: str = "test_name",
    role: str = "worker",
    env: Optional[Dict[str, str]] = None,
    *script_args: str,
    **script_kwargs: str,
) -> specs.AppDef:
    """
    Distributed data parallel style application (one role, multi-replica).
    The component is mostly used for running ddp-style applications via torchx cmd.

    Args:
        image: container image.
        entrypoint: script or binary to run within the image.
        resource: Optional resource identifier. The resource parameter
            gets ignored when running on the local scheduler.
        nnodes: Number of nodes.
        nproc_per_node: Number of processes per node.
        name: Name of the application.
        base_image: container base image (not required) .
        role: Name of the ddp role.
        script: Main script.
        env: Env variables.
        script_args: Script arguments.
        script_kwargs: Script arguments as dict.

    Returns:
        specs.AppDef: Torchx AppDef
    """

    args = list(script_args)
    for k, v in script_kwargs.items():
        args.append(f"--{k}")
        args.append(v)

    return torch_dist_component(
        name=name,
        role_name=role,
        image=image,
        base_image=base_image,
        entrypoint=entrypoint,
        resource=resource or specs.NULL_RESOURCE,
        args=args,
        env=env,
        nproc_per_node=nproc_per_node,
        nnodes=nnodes,
    )
