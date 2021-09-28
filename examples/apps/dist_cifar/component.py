# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Distributed Trainer Component
=============================

The module defines a cifar10 distributed trainer component.
The cifar10 distributed trainer uses `torch.distributed.run` to
spawn worker processes.
"""

from typing import Optional, Dict

import torchx.specs.api as torchx
from torchx.components.dist import ddp


def trainer(
    *script_args: str,
    image: str,
    resource: Optional[str] = None,
    base_image: Optional[str] = None,
    nnodes: int = 1,
    nproc_per_node: int = 1,
    rdzv_backend: str = "c10d",
    rdzv_endpoint: str = "localhost:29400",
    env: Optional[Dict[str, str]] = None,
) -> torchx.AppDef:
    """Defines the component for cifar10 distributed trainer.

    Args:
        image: image to run (e.g. foobar:latest)
        resource: string representation of the resource, registerred via entry_point.
            More info: https://pytorch.org/torchx/latest/configure.html
            Default is torchx.NULL_RESOURCE
        base_image: specifies the base image
        nnodes: number of nodes to run train on, default 1
        nproc_per_node: number of processes per node. Each process
            is assumed to use a separate GPU, default 1
        rdzv_backend: rendezvous backend to use, allowed values can be found at
            https://github.com/pytorch/pytorch/blob/master/torch/distributed/elastic/rendezvous/registry.py
            The default backend is `c10d`
        rdzv_endpoint: Controller endpoint. In case of rdzv_backend is etcd, this is a etcd
            endpoint, in case of c10d, this is the endpoint of one of the hosts.
            The default endpoint is `localhost:30001`
        env: env variables for the app
        script_args: Script arguments.

    Returns:
        specs.AppDef: Torchx AppDef
    """
    return ddp(
        *script_args,
        image=image,
        entrypoint="examples/apps/dist_cifar/train.py",
        rdzv_backend=rdzv_backend,
        rdzv_endpoint=rdzv_endpoint,
        resource=resource,
        nnodes=nnodes,
        nproc_per_node=nproc_per_node,
        base_image=base_image,
        name="cifar-trainer",
        role="worker",
        env=env,
    )
