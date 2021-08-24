#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""


"""

from torchx.runner import get_runner
from torchx.specs import Resource
from torchx.specs import named_resources

GiB: int = 1024


def p3_2xlarge() -> Resource:
    return Resource(
        # TODO(aivanou): Determine why the number of CPUs allowed to
        # be requested via volcano is N-1
        cpu=7,
        gpu=1,
        memMB=61 * GiB,
    )


def register_gpu_resource() -> None:
    res = p3_2xlarge()
    print(f"Registering resource: {res}")
    named_resources["p3_2xlarge"] = res


if __name__ == "__main__":
    register_gpu_resource()
    runner = get_runner("kubernetes")
