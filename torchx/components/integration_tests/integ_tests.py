# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Type
from types import ModuleType
from pyre_extensions import none_throws

from torchx.components.component_test_base import ComponentUtils
from torchx.specs import RunConfig, SchedulerBackend
from torchx.components.integration_tests.component_provider import ComponentProvider


@dataclass
class AppDefRun:
    provider_cls: Type[ComponentProvider]
    scheduler: SchedulerBackend
    image: str
    future: Future

    def __str__(self) -> str:
        msg = f"`{self.scheduler}`:`{self.image}`"
        return msg


class IntegComponentTest:
    def __init__(self) -> None:
        self._schedulers_args: Dict[str, RunConfig] = {
            "local": self._get_torchx_local_args(),
            "local_docker": self._get_torchx_local_docker_args(),
            "kubernetes": self._get_torchx_k8s_args(),
        }

    def run_components(
            self, module: ModuleType, scheduler_confs: List[Tuple[SchedulerBackend, str]], dryrun: bool = False
    ) -> None:
        component_providers_cls = self._get_component_providers(module)
        futures: List[AppDefRun] = []
        executor = ThreadPoolExecutor()
        for scheduler, image in scheduler_confs:
            sched_futures = self._run_component_providers(executor, component_providers_cls, scheduler, image, dryrun)
            futures += sched_futures
        self._wait_and_print_report(futures)

    def _get_component_providers(self, module: ModuleType) -> List[Type[ComponentProvider]]:
        providers = []
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if inspect.isclass(attr) and \
                    issubclass(attr, ComponentProvider) and not inspect.isabstract(attr):
                providers.append(attr)
        return providers

    def _run_component_providers(
            self,
            executor: ThreadPoolExecutor,
            component_providers_cls: List[Type[ComponentProvider]],
            scheduler: SchedulerBackend,
            image: str,
            dryrun: bool = False
    ) -> List[AppDefRun]:
        futures: List[AppDefRun] = []
        for provider_cls in component_providers_cls:
            future = executor.submit(self._run_component_provider, provider_cls, scheduler, image, dryrun)
            futures.append(AppDefRun(
                provider_cls=provider_cls,
                scheduler=scheduler,
                image=image,
                future=future
            ))
        return futures

    def _run_component_provider(self, component_provider_cls: Type[ComponentProvider],
                                scheduler: SchedulerBackend, image: str, dryrun: bool = False) -> str:
        provider: Optional[ComponentProvider] = None
        try:
            provider = none_throws(component_provider_cls(scheduler, image))
            app_def = none_throws(provider.get_app_def())
            scheduler_args = self._get_scheduler_args(scheduler)
            ComponentUtils.run_appdef_on_scheduler(app_def, scheduler, scheduler_args, dryrun)
            return app_def.name
        finally:
            if provider:
                provider.post_component_exec()

    def _wait_and_print_report(self, futures: List[AppDefRun]) -> None:
        app_runs: Dict[str, List[str]] = {}
        for future in futures:
            status, msg = self._get_app_def_run_status(future)
            if status not in app_runs:
                app_runs[status] = []
            app_runs[status].append(msg)
        for status, msgs in app_runs.items():
            print(f"\nStatus: `{status}`\n")
            print("\n".join(msgs))
            print("\n")

    def _get_app_def_run_status(self, app_def_run: AppDefRun) -> Tuple[str, str]:
        try:
            app_name = app_def_run.future.result()
            return "succeeded", f"`{app_name}`:`{app_def_run}`"
        except Exception as e:
            stack_trace_msg = traceback.format_exc().replace("\n", "\n  ")

            msg = (
                f"Failure while running: {app_def_run}, provider: `{app_def_run.provider_cls}`: {e}\n"
                f"Stacktrace: {stack_trace_msg}\n"
            )
            return "failed", msg

    def _get_scheduler_args(self, scheduler: SchedulerBackend) -> RunConfig:
        scheduler_args = self._schedulers_args.get(scheduler)
        if not scheduler_args:
            raise ValueError(f"No `scheduler args` registered for : {scheduler}")
        return scheduler_args

    def _get_torchx_k8s_args(self) -> RunConfig:
        cfg = RunConfig()
        cfg.set("namespace", "torchx-dev")
        cfg.set("queue", "default")
        return cfg

    def _get_torchx_local_args(self) -> RunConfig:
        cfg = RunConfig()
        cfg.set("image_type", "dir")
        return cfg

    def _get_torchx_local_docker_args(self) -> RunConfig:
        cfg = RunConfig()
        cfg.set("image_type", "docker")
        return cfg
