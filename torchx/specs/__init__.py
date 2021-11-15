#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This contains the TorchX AppDef and related component definitions. These are
used by components to define the apps which can then be launched via a TorchX
scheduler or pipeline adapter.
"""

from typing import Dict

import torchx.specs.named_resources_aws as aws_resources
from torchx.util.entrypoints import load_group

from .api import (  # noqa: F401 F403
    ALL,
    MISSING,
    NONE,
    NULL_RESOURCE,
    AppDef,
    AppDryRunInfo,
    AppHandle,
    AppState,
    AppStatus,
    CfgVal,
    InvalidRunConfigException,
    MalformedAppHandleException,
    ReplicaState,
    ReplicaStatus,
    Resource,
    RetryPolicy,
    Role,
    RoleStatus,
    SchedulerBackend,
    UnknownAppException,
    UnknownSchedulerException,
    from_function,
    get_argparse_param_type,
    get_type_name,
    is_terminal,
    macros,
    make_app_handle,
    parse_app_handle,
    runopt,
    runopts,
)


GiB: int = 1024


def _load_named_resources() -> Dict[str, Resource]:
    resource_methods = load_group("torchx.named_resources", default={})
    materialized_resources = {}
    default = {
        "aws_t3.medium": aws_resources.aws_t3_medium(),
        "aws_m5.2xlarge": aws_resources.aws_m5_2xlarge(),
        "aws_p3.2xlarge": aws_resources.aws_p3_2xlarge(),
        "aws_p3.8xlarge": aws_resources.aws_p3_8xlarge(),
    }
    for name, resource in default.items():
        materialized_resources[name] = resource
    for resource_name, resource_method in resource_methods.items():
        materialized_resources[resource_name] = resource_method()
    materialized_resources["NULL"] = NULL_RESOURCE
    return materialized_resources


named_resources: Dict[str, Resource] = _load_named_resources()


def get_named_resources(res: str) -> Resource:
    """
    Get resource object based on the string definition registered via entrypoints.txt.

    TorchX implements ``named_resource`` registration mechanism, which consists of
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
