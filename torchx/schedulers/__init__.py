#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

import torchx.schedulers.local_scheduler as local_scheduler
from torchx.schedulers.api import Scheduler
from torchx.specs.api import SchedulerBackend
from torchx.util.entrypoints import load_group


def get_schedulers(
    session_name: str,
    # pyre-ignore[2]
    **scheduler_params
) -> Dict[SchedulerBackend, Scheduler]:

    schedulers = load_group(
        "torchx.schedulers",
        default={
            "local": local_scheduler.create_scheduler,
            "default": local_scheduler.create_scheduler,
        },
        ignore_missing=True,
    )

    return {
        scheduler_backend: scheduler_factory_method(session_name, **scheduler_params)
        for scheduler_backend, scheduler_factory_method in schedulers.items()
    }
