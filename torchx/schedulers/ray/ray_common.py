# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Dict, List, Optional

TORCHX_RANK0_HOST: str = "TORCHX_RANK0_HOST"


@dataclass
class RayActor:
    """Describes an actor (a.k.a. worker/replica in TorchX terms)."""

    name: str
    command: List[str]
    env: Dict[str, str] = field(default_factory=dict)
    num_cpus: int = 1
    num_gpus: int = 0
    nnodes_rep: Optional[str] = None


@dataclass
class RayResult:
    id: str


class TaskCompleted(RayResult):
    def __init__(self, id: str) -> None:
        super().__init__(id=id)


class CommandActorScheduled(RayResult):
    def __init__(self, id: str) -> None:
        super().__init__(id=id)
