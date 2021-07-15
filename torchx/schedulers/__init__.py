#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torchx.schedulers.kubernetes_scheduler as kubernetes_scheduler
import torchx.schedulers.local_scheduler as local_scheduler
import torchx.schedulers.slurm_scheduler as slurm_scheduler
from torchx.schedulers.api import Scheduler
from torchx.specs.api import SchedulerBackend
from torchx.util.entrypoints import load_group
from typing_extensions import Protocol


class SchedulerFactory(Protocol):
    def __call__(self, session_name: str, **kwargs: object) -> Scheduler:
        ...


def get_schedulers(
    session_name: str, **scheduler_params: object
) -> Dict[SchedulerBackend, Scheduler]:
    default_schedulers: Dict[str, SchedulerFactory] = {
        "local": local_scheduler.create_scheduler,
        "default": local_scheduler.create_scheduler,
        "slurm": slurm_scheduler.create_scheduler,
        "kubernetes": kubernetes_scheduler.create_scheduler,
    }

    schedulers = load_group(
        "torchx.schedulers",
        default=default_schedulers,
        ignore_missing=True,
    )

    return {
        scheduler_backend: scheduler_factory_method(session_name, **scheduler_params)
        for scheduler_backend, scheduler_factory_method in schedulers.items()
    }
