#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
import warnings
from typing import Callable, Dict, List, Mapping

from torchx.specs.api import DeviceMount


def efa_to_devicemounts(num_devices: int) -> List[DeviceMount]:
    device_mounts = []
    for device_index in range(0, num_devices):
        device_mounts.append(
            DeviceMount(
                src_path="/dev/infiniband/uverbs" + str(device_index),
                dst_path="/dev/infiniband/uverbs" + str(device_index),
            )
        )
    return device_mounts


DEVICES: Mapping[str, Callable[[int], List[DeviceMount]]] = {
    "vpc.amazonaws.com/efa": efa_to_devicemounts,
}


def get_device_mounts(devices: Dict[str, int]) -> List[DeviceMount]:
    """
    Takes in a list of named devices/quantities, and returns a list of DeviceMount objects
    based on the mappings defined in DEVICES
    """
    device_mounts = []
    for device_name, num_devices in devices.items():
        if device_name not in DEVICES:
            warnings.warn(f"Could not find named device: {device_name}")
            continue
        device_mounts += DEVICES[device_name](num_devices)
    return device_mounts
