# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
These components are meant to be used as convenience methods when constructing
other components. Many methods in the base component library are factory methods
for ``Role``, ``Container``, and ``Resources`` that are hooked up to
TorchX's configurable extension points.
"""
from typing import Any, Dict, List, Optional

from torchx.specs.api import NULL_RESOURCE, Resource, RetryPolicy, Role
from torchx.util.entrypoints import load

from .roles import create_torch_dist_role  # noqa: F401 F403


def named_resource(name: str) -> Resource:
    # TODO <PLACEHOLDER> read instance types and resource mappings from entrypoints
    return NULL_RESOURCE


def torch_dist_role(
    name: str,
    image: str,
    entrypoint: str,
    resource: Resource = NULL_RESOURCE,
    base_image: Optional[str] = None,
    script_args: Optional[List[str]] = None,
    script_envs: Optional[Dict[str, str]] = None,
    num_replicas: int = 1,
    max_retries: int = 0,
    port_map: Optional[Dict[str, int]] = None,
    retry_policy: RetryPolicy = RetryPolicy.APPLICATION,
    **launch_kwargs: Any,
) -> Role:
    """
    A ``Role`` for which the user provided ``entrypoint`` is executed with the
        torchelastic agent (in the container). Note that the torchelastic agent
        invokes multiple copies of ``entrypoint``.

    The method will try to search factory method that is registerred via entrypoints.
    If no group or role found, the default ``torchx.components.base.role.create_torch_dist_role``
    will be used.

    For more information see ``torchx.components.base.roles``

    Usage:

    ::

     # nnodes and nproc_per_node correspond to the ``torch.distributed.launch`` arguments. More
     # info about available arguments: https://pytorch.org/docs/stable/distributed.html#launch-utility
     trainer = torch_dist_role("trainer",container, entrypoint="trainer.py",.., nnodes=2, nproc_per_node=4)

    Args:
        name: Name of the role
        container: Container
        entrypoint: Script or binary to launch
        script_args: Arguments to the script
        script_envs: Env. variables to the worker
        num_replicas: Number of replicas
        max_retries: Number of retries
        retry_policy: ``torchx.specs.api.RetryPolicy``
        launch_kwargs: ``torch.distributed.launch`` arguments.

    Returns:
        Torchx role

    """
    dist_role_factory = load(
        "torchx.base",
        "dist_role",
        default=create_torch_dist_role,
    )

    return dist_role_factory(
        name,
        image,
        entrypoint,
        resource,
        base_image,
        script_args,
        script_envs,
        num_replicas,
        max_retries,
        port_map or {},
        retry_policy,
        **launch_kwargs,
    )
