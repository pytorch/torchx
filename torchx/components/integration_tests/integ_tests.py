# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Optional, Callable

import torchx.components.utils as utils_components
from torchx.components.component_test_base import ComponentTestCase
from torchx.specs import RunConfig, AppDef, SchedulerBackend


@dataclass
class AppDefRun:
    app_name: str
    scheduler: str
    future: Future


class IntegComponentTest:
    def __init__(self) -> None:
        self.component_test_case = ComponentTestCase()
        self._schedulers_args: Dict[str, RunConfig] = {
            "local": self._get_torchx_local_args(),
            "kubernetes": self._get_torchx_k8s_args(),
        }

    def run_builtin_components(
        self, image: str, schedulers: List[SchedulerBackend], dryrun: bool = False
    ) -> None:
        return self.run_components(
            AppDefProvider().get_app_defs, image, schedulers, dryrun
        )

    def run_components(
        self,
        components_fetcher: Callable[[str, SchedulerBackend], List[AppDef]],
        image: str,
        schedulers: List[SchedulerBackend],
        dryrun: bool = False,
    ) -> None:
        with ThreadPoolExecutor() as executor:
            futures: List[AppDefRun] = []
            for scheduler in schedulers:
                app_defs = components_fetcher(image, scheduler)
                for app_def in app_defs:
                    future = self.run_appdef_on_scheduler(
                        executor, app_def, scheduler, dryrun
                    )
                    futures.append(AppDefRun(app_def.name, scheduler, future))
            report = self._wait_and_construct_report(futures)
            print(report)

    def _wait_and_construct_report(self, futures: List[AppDefRun]) -> str:
        return "\n".join(
            self._get_app_def_run_status(app_def_run) for app_def_run in futures
        )

    def _get_app_def_run_status(self, app_def_run: AppDefRun) -> str:
        try:
            app_def_run.future.result()
            return f"`{app_def_run.app_name}`:`{app_def_run.scheduler}` succeeded"
        except Exception as e:
            return f"`{app_def_run.app_name}`:`{app_def_run.scheduler}` failed with error: {e}"

    def run_appdef_on_scheduler(
        self,
        executor: ThreadPoolExecutor,
        app_def: AppDef,
        scheduler: SchedulerBackend,
        dryrun: bool = False,
    ) -> Future:
        scheduler_args = self._schedulers_args.get(scheduler)
        if not scheduler_args:
            raise ValueError(f"Unknown scheduler: {scheduler}")
        return executor.submit(
            self.component_test_case.run_appdef_on_scheduler,
            app_def=app_def,
            scheduler=scheduler,
            scheduler_cfg=scheduler_args,
            dryrun=dryrun,
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


class AppDefLoader:
    @classmethod
    def load_apps_defs(
        cls, obj: object, image: str, scheduler: SchedulerBackend
    ) -> List[AppDef]:
        methods = []
        start_pattern = "_get_app_def"
        for attr in dir(obj):
            if attr.startswith(start_pattern):
                methods.append(attr)
        app_defs = []
        for method in methods:
            app_def_fn = getattr(obj, method)
            app_defs.append(app_def_fn(image, scheduler))
        return app_defs


class AppDefProvider:
    def get_app_defs(self, image: str, scheduler: SchedulerBackend) -> List[AppDef]:
        app_defs = AppDefLoader.load_apps_defs(self, image, scheduler)
        return [app_def for app_def in app_defs if app_def]

    def _get_app_def_echo(
        self, image: str, scheduler: SchedulerBackend
    ) -> Optional[AppDef]:
        return utils_components.echo()

    def _get_app_def_sh(
        self, image: str, scheduler: SchedulerBackend
    ) -> Optional[AppDef]:
        return utils_components.sh("echo", "test")

    def _get_app_def_booth(
        self, image: str, scheduler: SchedulerBackend
    ) -> Optional[AppDef]:
        return utils_components.booth(1, 2)
