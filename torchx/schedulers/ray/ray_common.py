# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class RayActor:
    """Describes an actor (a.k.a. role in TorchX terms).

    Attributes:
        name:
            The name of the actor.
        command:
            The command that the actor should run as a subprocess.
        env:
            The environment variables to set before executing the command.
        num_replicas:
            The number of replicas (i.e. Ray actors) to run.
        num_cpus:
            The number of CPUs to allocate.
        num_gpus:
            The number of GPUs to allocate.
    """

    name: str
    command: str
    env: Dict[str, str] = field(default_factory=dict)
    num_replicas: int = 1
    num_cpus: int = 1
    num_gpus: int = 0
    # TODO: memory_size, max_retries, retry_policy
