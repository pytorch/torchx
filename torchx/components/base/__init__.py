# Copyright (c) Meta Platforms, Inc. and affiliates.
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
from typing import Any, Dict, List, Optional, Union

from torchx.specs import named_resources
from torchx.specs.api import NULL_RESOURCE, Resource, RetryPolicy, Role
from torchx.util.entrypoints import load

from .roles import create_torch_dist_role


def _resolve_resource(resource: Union[str, Resource]) -> Resource:
    if isinstance(resource, Resource):
        return resource
    else:
        return named_resources[resource]


def torch_dist_role(
    name: str,
    image: str,
    entrypoint: str,
    resource: Union[str, Resource] = NULL_RESOURCE,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    num_replicas: int = 1,
    max_retries: int = 0,
    port_map: Optional[Dict[str, int]] = None,
    retry_policy: RetryPolicy = RetryPolicy.APPLICATION,
    **launch_kwargs: Any,
) -> Role:
    """
    .. warning:: This method is deprecated and will be removed in future versions.
                 Instead use :py:func:`torchx.components.dist.ddp` as a builtin,
                 or prefer to use `torch.distributed.run <https://pytorch.org/docs/stable/elastic/run.html>`_
                 directly by setting your AppDef's ``entrypoint = python`` and
                 ``args = ["-m", "torch.distributed.run", ...]``.

    A ``Role`` for which the user provided ``entrypoint`` is executed with the
        torchelastic agent (in the container). Note that the torchelastic agent
        invokes multiple copies of ``entrypoint``.

    The method will try to search factory method that is registered via entrypoints.
    If no group or role found, the default ``torchx.components.base.role.create_torch_dist_role``
    will be used.

    For more information see ``torchx.components.base.roles``

    Usage:

    ::

     # nproc_per_node correspond to the ``torch.distributed.run`` arguments. More
     # info about available arguments: https://pytorch.org/docs/stable/elastic/run.html
     trainer = torch_dist_role("trainer",container, entrypoint="trainer.py",.., nproc_per_node=4)


    .. warning:: Users can provide ``nnodes`` parameter, that is used by the ``torch.distibuted.run``
                  If users do not provide ``nnodes`` parameter, the ``nnodes`` will be automatically set
                  to ``num_replicas``.


    Args:
        name: Name of the role
        image: Image of the role
        entrypoint: Script or binary to launch
        resource: Resource specs that define the container properties. Predefined resources
            are supported as str arguments.
        args: Arguments to the script
        env: Env. variables to the worker
        num_replicas: Number of replicas
        max_retries: Number of retries
        retry_policy: ``torchx.specs.api.RetryPolicy``
        launch_kwargs: ``torch.distributed.launch`` arguments.

    Returns:
        TorchX role

    """
    dist_role_factory = load(
        "torchx.base",
        "dist_role",
        default=create_torch_dist_role,
    )

    resource = _resolve_resource(resource)

    return dist_role_factory(
        name,
        image,
        entrypoint,
        resource,
        args,
        env,
        num_replicas,
        max_retries,
        port_map or {},
        retry_policy,
        **launch_kwargs,
    )
