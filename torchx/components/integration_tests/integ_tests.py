# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
from typing import List

import examples.apps.lightning_classy_vision.component as cv_component
import examples.apps.datapreproc.component as dp_component
import torchx.components.dist as dist
from torchx.components.component_test_base import ComponentTestCase
from torchx.runner import get_runner
from torchx.specs import RunConfig, AppDef, SchedulerBackend


class IntegComponentTest:
    def __init__(self):
        self.component_test_case = ComponentTestCase()
        self._schedulers_args = {
            "local": self._get_torchx_local_args(),
            "kubernetes": self._get_torchx_k8s_args()
        }

    def run_builtin_components(
            self, image: str, schedulers: List[SchedulerBackend], dryrun: bool = False
    ) -> None:
        app_defs = self._get_builtin_app_defs(image)
        for app_def in app_defs:
            self.run_app_def_on_schedulers(app_def, schedulers, dryrun)

    def run_app_def_on_schedulers(self, app_def: AppDef, schedulers: List[SchedulerBackend],
                                  dryrun: bool = False) -> None:
        for scheduler in schedulers:
            scheduler_args = self._schedulers_args.get(scheduler)
            if not scheduler_args:
                raise ValueError(f"Unknown scheduler: {scheduler}")
            self.component_test_case.run_appdef_on_scheduler(app_def, scheduler, scheduler_args, dryrun)

    def _get_builtin_app_defs(self, image: str) -> List[AppDef]:
        cv_trainer = self._get_cv_app_def(image)
        dp_app = self._get_dp_app_def(image)
        return [cv_trainer, dp_app]

    def _run_app_def(self, app_def, scheduler, scheduler_args):
        runner = get_runner("test-runner")
        app_handle = runner.run(app_def, scheduler, scheduler_args)
        print(app_handle)
        print(runner.wait(app_handle))

    def _get_cv_app_def(self, image: str) -> AppDef:
        return cv_component.trainer(
            image=image,
            output_path="/tmp",
            skip_export=True,
            log_path="/tmp",
        )

    def _get_dp_app_def(self, image: str) -> AppDef:
        return dp_component.data_preproc(
            image=image,
            output_path="/tmp",
        )

    def _get_torchx_k8s_args(self) -> RunConfig:
        cfg = RunConfig()
        cfg.set("namespace", "torchx-dev")
        cfg.set("queue", "default")
        return cfg

    def _get_torchx_local_args(self) -> RunConfig:
        cfg = RunConfig()
        cfg.set("image_type", "docker")
        return cfg
