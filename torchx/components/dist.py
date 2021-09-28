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

from typing import Dict, Optional, Any

import torchx.specs as specs
from torchx.components.base import torch_dist_role


def ddp(
    *script_args: str,
    image: str,
    entrypoint: str,
    rdzv_backend: Optional[str] = None,
    rdzv_endpoint: Optional[str] = None,
    resource: Optional[str] = None,
    nnodes: int = 1,
    nproc_per_node: int = 1,
    base_image: Optional[str] = None,
    name: str = "test-name",
    role: str = "worker",
    env: Optional[Dict[str, str]] = None,
) -> specs.AppDef:
    """
    Distributed data parallel style application (one role, multi-replica).

    This uses `Torch Elastic
    <https://pytorch.org/docs/stable/distributed.elastic.html>`_ to manage the
    distributed workers.

    Args:
        script_args: Script arguments.
        image: container image.
        entrypoint: script or binary to run within the image.
        rdzv_backend: rendezvous backend to use, allowed values can be found at
            https://github.com/pytorch/pytorch/blob/master/torch/distributed/elastic/rendezvous/registry.py
        rdzv_endpoint: Controller endpoint. In case of rdzv_backend is etcd, this is a etcd
            endpoint, in case of c10d, this is the endpoint of one of the hosts.
        resource: Optional named resource identifier. The resource parameter
            gets ignored when running on the local scheduler.
        nnodes: Number of nodes.
        nproc_per_node: Number of processes per node.
        name: Name of the application.
        base_image: container base image (not required) .
        role: Name of the ddp role.
        env: Env variables.

    Returns:
        specs.AppDef: Torchx AppDef
    """

    launch_kwargs: Dict[str, Any] = {
        "nnodes": nnodes,
        "nproc_per_node": nproc_per_node,
        "max_restarts": 0,
    }
    if rdzv_backend:
        launch_kwargs["rdzv_backend"] = rdzv_backend
    if rdzv_endpoint:
        launch_kwargs["rdzv_endpoint"] = rdzv_endpoint

    retry_policy: specs.RetryPolicy = specs.RetryPolicy.APPLICATION

    ddp_role = torch_dist_role(
        name=role,
        image=image,
        entrypoint=entrypoint,
        resource=resource or specs.NULL_RESOURCE,
        base_image=base_image,
        args=list(script_args),
        env=env,
        num_replicas=nnodes,
        max_retries=0,
        retry_policy=retry_policy,
        port_map=None,
        **launch_kwargs,
    )

    return specs.AppDef(name, roles=[ddp_role])
