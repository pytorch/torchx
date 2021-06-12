#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .api import *  # noqa: F401 F403
from torchx.util.entrypoints import load


def named_resource(resource: str) -> Resource:
    """
    Gets resource object based on the string definition registered via entrypoints.txt.
    """
    try:
        resource_fn = load("torchx.resources", name=resource.lower())
    except KeyError:
        raise ValueError(
            f"Resource: {resource} is not registered, check entrypoints.txt file."
        )
    return resource_fn()
