# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Union

from torchx.components.base import torch_dist_role
from torchx.specs import api


def torch_dist_component(
    name: str,
    image: str,
    entrypoint: str,
    nnodes: int = 1,
    nproc_per_node: int = 1,
    role_name: str = "worker",
    base_image: Optional[str] = None,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    resource: Union[str, api.Resource] = api.NULL_RESOURCE,
) -> api.AppDef:
    """
    Generic component that uses launches workers via torch.distributed.run

    >>> from torchx.components.base.torch_dist_component import torch_dist_component
    >>> torch_dist_component(
    ...     name="datapreproc",
    ...     image="pytorch/pytorch:latest",
    ...     entrypoint="python3",
    ...     args=["--version"],
    ...     env={
    ...         "FOO": "bar",
    ...     },
    ... )
    AppDef(name='datapreproc', ...)


    """

    env = env or {}
    args = args or []

    role = torch_dist_role(
        name=role_name,
        image=image,
        base_image=base_image,
        entrypoint=entrypoint,
        resource=resource,
        env=env,
        args=args,
        num_replicas=nnodes,
        nnodes=nnodes,
        nproc_per_node=nproc_per_node,
    )

    return api.AppDef(
        name=name,
        roles=[role],
    )
