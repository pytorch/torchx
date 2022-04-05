#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import Mapping, Tuple, Callable

from torchx.specs.api import DeviceMount

DEVICES: Mapping[str, Callable] = {
    "vpc.amazonaws.com/efa": lambda device_index: DeviceMount(
        src_path="/dev/infiniband/uverbs" + str(device_index),
        dst_path="/dev/infiniband/uverbs" + str(device_index),
    ),
    "vpc.amazonaws.com/neuron": lambda device_index: DeviceMount(
        src_path="/dev/neuron/uverbs" + str(device_index),
        dst_path="/dev/neuron/uverbs" + str(device_index),
    ),
}


def get_device_mounts(devices: Tuple[str, int]) -> list[DeviceMount]:
    """
    Takes in a list of named devices/quantities, and returns a list of DeviceMount objects
    based on the mappings defined in DEVICES
    """
    device_mounts = []
    for device in devices:
        [device_name, num_devices] = device
        if device_name not in DEVICES:
            continue
        device_mounts+=list(map(DEVICES[device_name], range(0, num_devices)))
    return device_mounts
