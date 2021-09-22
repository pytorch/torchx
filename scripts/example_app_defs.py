#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
App Defs for integration tests.
"""

from typing import Optional, List

import examples.apps.datapreproc.component as dp_component
import examples.apps.dist_cifar.component as dist_cifar_component
import examples.apps.lightning_classy_vision.component as cv_component
from torchx.components.integration_tests.integ_tests import AppDefLoader
from torchx.specs import AppDef, SchedulerBackend


class ExamplesAppDefProvider:
    def get_example_app_defs(
            self, image: str, scheduler: SchedulerBackend
    ) -> List[AppDef]:
        app_defs: List[AppDef] = AppDefLoader.load_apps_defs(self, image, scheduler)
        return [app_def for app_def in app_defs if app_def]

    def _get_app_def_cv(
            self, image: str, scheduler: SchedulerBackend
    ) -> Optional[AppDef]:
        return cv_component.trainer(
            image=image,
            output_path="/tmp",
            skip_export=True,
            log_path="/tmp",
        )

    def _get_app_def_dist_cifar(
            self, image: str, scheduler: SchedulerBackend
    ) -> Optional[AppDef]:
        args = ["--output_path", "/tmp", "--dryrun"]
        return dist_cifar_component.trainer(
            *args,
            image=image,
        )

    def _get_app_def_dp(
            self, image: str, scheduler: SchedulerBackend
    ) -> Optional[AppDef]:
        return dp_component.data_preproc(
            image=image,
            output_path="/tmp",
            dryrun=True,
        )
