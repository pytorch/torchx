# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile

import examples.apps.lightning_classy_vision.component as cv_component
import torchx.components.dist as dist
from torchx.components.component_test_base import ComponentTestCase
from torchx.runner import get_runner
from torchx.specs import RunConfig, AppDef


class IntegComponentTest:
    def run_app_def_on_schedulers(self):
        pass

    def run_on_k8s(self, image: str):
        cfg = self._get_torchx_k8s_args()
        return self.run_builtin_components(image, "kubernetes", cfg)

    def run_on_local(self, image: str):
        cfg = self._get_torchx_local_args()
        return self.run_builtin_components(image, "local", cfg)

    def run_builtin_components(
            self, image: str, scheduler: str, scheduler_args: RunConfig
    ) -> None:
        cv_trainer = self._get_cv_app_def(image)
        self._run_app_def(cv_trainer, scheduler, scheduler_args)

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

    def _get_torchx_k8s_args(self) -> RunConfig:
        cfg = RunConfig()
        cfg.set("namespace", "torchx-dev")
        cfg.set("queue", "default")
        return cfg

    def _get_torchx_local_args(self) -> RunConfig:
        cfg = RunConfig()
        cfg.set("img_type", "docker")
        return cfg
