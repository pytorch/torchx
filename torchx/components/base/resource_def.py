# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

from torchx.specs.api import Resource, NULL_RESOURCE

GiB: int = 1024


# The alias is used for testing when the resource requirements do not matter
IGNORED: Resource = NULL_RESOURCE


def list_resources() -> Dict[str, Resource]:
    return {
        # aws t4g.large
        "cpu_x_2": Resource(cpu=2, gpu=0, memMB=8 * GiB),
        # aws t4g.xlarge
        "cpu_x_4": Resource(cpu=4, gpu=0, memMB=16 * GiB),
        # aws t4g.2xlarge
        "cpu_x_8": Resource(cpu=8, gpu=0, memMB=32 * GiB),
        # aws p3.Xxlarge
        "gpu_x_1": Resource(cpu=8, gpu=1, memMB=61 * GiB),
        "gpu_x_2": Resource(cpu=16, gpu=2, memMB=122 * GiB),
        "gpu_x_4": Resource(cpu=32, gpu=4, memMB=244 * GiB),
        "gpu_x_8": Resource(cpu=64, gpu=8, memMB=488 * GiB),
        "IGNORED": IGNORED,
    }
