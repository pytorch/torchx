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
from torchx.specs.finder import get_components, _Component
import inspect


@dataclass
class AppDefRun:
    app_name: str
    scheduler: str
    future: Future


class IntegComponentTest:
    def __init__(self) -> None:
        self.component_test_case = ComponentTestCase()
        self._schedulers_args: Dict[str, RunConfig] = {
            "local_docker": self._get_torchx_local_args(),
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
                    # futures.append(AppDefRun(app_def.name, scheduler, future))
            # report = self._wait_and_construct_report(futures)
            # print(report)

    def _wait_and_construct_report(self, futures: List[AppDefRun]) -> str:
        return "\n".join(
            self._get_app_def_run_status(app_def_run) for app_def_run in futures
        )

    def _get_app_def_run_status(self, app_def_run: AppDefRun) -> str:
        try:
            app_def_run.future.result()
            return f"`{app_def_run.app_name}`:`{app_def_run.scheduler}` succeeded"
        except Exception as e:
            return (f"`{app_def_run.app_name}`:`{app_def_run.scheduler}` failed with error: {e}."
                    "Check logs for root cause")

    def run_appdef_on_scheduler(
            self,
            executor: ThreadPoolExecutor,
            app_def: AppDef,
            scheduler: SchedulerBackend,
            dryrun: bool = False,
    ) -> Future:
        scheduler_args = self._schedulers_args.get(scheduler)
        if not scheduler_args:
            raise ValueError(f"No `scheduler args` registered for : {scheduler}")
        return self.component_test_case.run_appdef_on_scheduler(
            app_def=app_def,
            scheduler=scheduler,
            scheduler_cfg=scheduler_args,
            dryrun=dryrun,
        )

        # return executor.submit(
        #     self.component_test_case.run_appdef_on_scheduler,
        #     app_def=app_def,
        #     scheduler=scheduler,
        #     scheduler_cfg=scheduler_args,
        #     dryrun=dryrun,
        # )

    def _get_torchx_k8s_args(self) -> RunConfig:
        cfg = RunConfig()
        cfg.set("namespace", "torchx-dev")
        cfg.set("queue", "default")
        return cfg

    def _get_torchx_local_args(self) -> RunConfig:
        cfg = RunConfig()
        cfg.set("image_type", "docker")
        return cfg


class AppDefProvider:
    def get_app_defs(self, image: str, scheduler: SchedulerBackend) -> List[AppDef]:
        components = get_components().values()
        app_defs = [self._instantiate_component(component, image, scheduler) for component in components]
        return app_defs

    def _instantiate_component(self, component: _Component, image: str, scheduler: SchedulerBackend) -> AppDef:
        return None

    def _app_def_overrides(self, component: _Component) -> Optional[AppDef]:
        overrides = {}

    def _app_def_copy(self, image: str, component: _Component) -> AppDef:
        return component.fn(
            src="examples/apps/dist_cifar.py",
            dst="examples/apps/dist_cifar.py.copy",
        )

    def _app_def_ddp(self, image: str, component: _Component) -> AppDef:
        args = ["--dryrun"]
        return component.fn(
            *args,
            entrypoint="examples/apps/dist_cifar/train.py",
            image=image,
        )

    def _get_int(self):
        return 0

    def _get_str(self):
        return "test"

    def _get_float(self):
        return 1.0

    def _get_bool(self):
        return True

    def _get_list(self):
        return []

    def _get_dict(self):
        return {}
