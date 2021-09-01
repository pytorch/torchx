#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

from torchx.util.entrypoints import load_group

from .api import (  # noqa: F401 F403
    SchedulerBackend,
    Resource,
    NULL_RESOURCE,
    ALL,
    MISSING,
    NONE,
    macros,
    RetryPolicy,
    Role,
    AppDef,
    AppState,
    is_terminal,
    ReplicaStatus,
    ReplicaState,
    RoleStatus,
    AppStatus,
    ConfigValue,
    RunConfig,
    AppDryRunInfo,
    get_type_name,
    runopts,
    InvalidRunConfigException,
    MalformedAppHandleException,
    UnknownSchedulerException,
    AppHandle,
    UnknownAppException,
    make_app_handle,
    parse_app_handle,
    get_argparse_param_type,
    from_function,
)

GiB: int = 1024


def _load_named_resources() -> Dict[str, Resource]:
    resource_methods = load_group("torchx.named_resources", default={})
    materialized_resources = {}
    for resource_name, resource_method in resource_methods.items():
        materialized_resources[resource_name] = resource_method()
    materialized_resources["NULL"] = NULL_RESOURCE
    return materialized_resources


named_resources: Dict[str, Resource] = _load_named_resources()


def get_named_resources(res: str) -> Resource:
    """
    Get resource object based on the string definition registered via entrypoints.txt.

    Torchx implements ``named_resource`` registration mechanism, which consists of
    the following steps:

    1. Create a module and define your resource retrieval function:

    .. code-block:: python

     # my_module.resources
     from typing import Dict
     from torchx.specs import Resource

     def gpu_x_1() -> Dict[str, Resource]:
         return Resource(cpu=2, memMB=64 * 1024, gpu = 2)

    2. Register resource retrieval in the entrypoints section:

    ::

     [torchx.named_resources]
     gpu_x_1 = my_module.resources:gpu_x_1

    The ``gpu_x_1`` can be used as string argument to this function:

    ::

     from torchx.specs import named_resources
     resource = named_resources["gpu_x_1"]

    """
    return named_resources[res]
