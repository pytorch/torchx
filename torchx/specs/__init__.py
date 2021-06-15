#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .api import *  # noqa: F401 F403
from torchx.util.entrypoints import load

GiB: int = 1024


def named_resource(resource: str) -> Resource:
    """
    Gets resource object based on the string definition registered via entrypoints.txt.

    Torchx implements named resource registration mechanism, which consists of
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

     [torchx.schedulers]
     gpu_x_1 = my_module.resources:gpu_x_1

    The ``gpu_x_1`` can be used as string argument to this function:

    ::

     resource = named_resource("gpu_x_1")

    """
    try:
        resource_fn = load("torchx.resources", name=resource.lower())
    except KeyError:
        raise ValueError(
            f"Resource: {resource} is not registered, check entrypoints.txt file."
        )
    return resource_fn()
