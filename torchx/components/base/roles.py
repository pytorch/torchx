# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import field
from typing import Any, Dict, List, Optional

from torchx.specs.api import NULL_RESOURCE, Resource, RetryPolicy, Role, macros


logger: logging.Logger = logging.getLogger(__name__)


def create_torch_dist_role(
    name: str,
    image: str,
    entrypoint: str,
    resource: Resource = NULL_RESOURCE,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
    num_replicas: int = 1,
    max_retries: int = 0,
    port_map: Dict[str, int] = field(default_factory=dict),
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

    For more information about torchelastic see
    `docs <https://pytorch.org/docs/stable/distributed.elastic.html>`_.

    .. important:: It is the responsibility of the user to ensure that the
                   role's image includes torchelastic. Since TorchX has no
                   control over the build process of the image, it cannot
                   automatically include torchelastic in the role's image.

    The following example launches 2 ``replicas`` (nodes) of an elastic ``my_train_script.py``
    that is allowed to scale between 2 to 4 nodes. Each node runs 8 workers which are allowed
    to fail and restart a maximum of 3 times.

    >>> from torchx.components.base.roles import create_torch_dist_role
    >>> from torchx.specs.api import NULL_RESOURCE
    >>> elastic_trainer = create_torch_dist_role(
    ...     name="trainer",
    ...     image="<NONE>",
    ...     resource=NULL_RESOURCE,
    ...     entrypoint="my_train_script.py",
    ...     args=["--script_arg", "foo", "--another_arg", "bar"],
    ...     num_replicas=4, max_retries=1,
    ...     nproc_per_node=8, max_restarts=3)
    ... # effectively runs:
    ... #    python -m torch.distributed.launch
    ... #        --nproc_per_node 8
    ... #        --nnodes 4
    ... #        --max_restarts 3
    ... #        my_train_script.py --script_arg foo --another_arg bar
    >>> elastic_trainer
    Role(name='trainer', ...)


    Args:
        name: Name of the role
        entrypoint: User binary or python script that will be launched.
        resource: Resource that is requested by scheduler
        args: User provided arguments
        env: Env. variables that will be set on worker process that runs entrypoint
        num_replicas: Number of role replicas to run
        max_retries: Max number of retries
        port_map: Port mapping for the role
        retry_policy: Retry policy that is applied to the role
        launch_kwargs: kwarg style launch arguments that will be used to launch torchelastic agent.

    Returns:
        Role object that launches user entrypoint via the torchelastic as proxy

    """
    args = args or []
    env = env or {}

    entrypoint_override = "python"
    torch_run_args: List[str] = ["-m", "torch.distributed.run"]

    launch_kwargs.setdefault("rdzv_backend", "c10d")
    launch_kwargs.setdefault("rdzv_endpoint", "localhost:29500")
    launch_kwargs.setdefault("rdzv_id", macros.app_id)
    launch_kwargs.setdefault("role", name)

    if "nnodes" not in launch_kwargs:
        launch_kwargs["nnodes"] = num_replicas

    for (arg, val) in launch_kwargs.items():
        if isinstance(val, bool):
            # treat boolean kwarg as a flag
            if val:
                torch_run_args += [f"--{arg}"]
        else:
            torch_run_args += [f"--{arg}", str(val)]

    args = [*torch_run_args, entrypoint, *args]
    return Role(
        name,
        image=image,
        entrypoint=entrypoint_override,
        args=args,
        env=env,
        num_replicas=num_replicas,
        retry_policy=retry_policy,
        max_retries=max_retries,
        resource=resource,
        port_map=port_map,
    )
