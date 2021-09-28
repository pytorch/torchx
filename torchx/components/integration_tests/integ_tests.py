# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from types import ModuleType
from typing import List, Dict, Optional, Tuple, Type, Callable, cast

from pyre_extensions import none_throws
from torchx.components.component_test_base import ComponentUtils
from torchx.components.integration_tests.component_provider import ComponentProvider
from torchx.specs import RunConfig, SchedulerBackend


@dataclass
class SchedulerInfo:
    name: SchedulerBackend
    image: str
    runconfig: RunConfig


@dataclass
class AppDefRun:
    provider_cls: Type[ComponentProvider]
    scheduler_info: SchedulerInfo
    future: Future

    def __str__(self) -> str:
        msg = f"`{self.scheduler_info.name}`:`{self.scheduler_info.image}`"
        return msg


class IntegComponentTest:
    """
    Class contains common methods for executing e2e integration tests of components on different schedulers.
    The providers of `AppDef` must implement `torchx.components.integration_tests.component_provider.ComponentProvider`.

    >>> import torchx.components.integration_tests.integ_tests as integ_tests
    >>> from torchx.components.integration_tests.component_provider import ComponentProvider
    >>> class MyComponentProvider(ComponentProvider):
    ...     def get_app_def(self):
    ...         return AppDef(...)
    >>> IntegComponentTest(...).run_component(my_module)

    #TODO(aivanou): implement `run_component` method that allows running a single component provider
    """

    def __init__(self, timeout: int = 300) -> None:
        self._timeout = timeout

    def run_components(
        self,
        module: ModuleType,
        scheduler_infos: List[SchedulerInfo],
        dryrun: bool = False,
    ) -> None:
        component_providers_cls = self._get_component_providers(module)
        futures: List[AppDefRun] = []
        executor = ThreadPoolExecutor()
        for scheduler_info in scheduler_infos:
            sched_futures = self._run_component_providers(
                executor, component_providers_cls, scheduler_info, dryrun
            )
            futures += sched_futures
        self._wait_and_print_report(futures)

    def _get_component_providers(
        self, module: ModuleType
    ) -> List[Type[ComponentProvider]]:
        providers = []
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if (
                inspect.isclass(attr)
                and issubclass(attr, ComponentProvider)
                and not inspect.isabstract(attr)
            ):
                providers.append(attr)
        return providers

    def _run_component_providers(
        self,
        executor: ThreadPoolExecutor,
        component_providers_cls: List[Type[ComponentProvider]],
        scheduler_info: SchedulerInfo,
        dryrun: bool = False,
    ) -> List[AppDefRun]:
        futures: List[AppDefRun] = []
        for provider_cls in component_providers_cls:
            future = executor.submit(
                self._run_component_provider, provider_cls, scheduler_info, dryrun
            )
            futures.append(
                AppDefRun(
                    provider_cls=provider_cls,
                    scheduler_info=scheduler_info,
                    future=future,
                )
            )
        return futures

    def _run_component_provider(
        self,
        component_provider_cls: Type[ComponentProvider],
        scheduler_info: SchedulerInfo,
        dryrun: bool = False,
    ) -> str:
        provider: Optional[ComponentProvider] = None
        try:
            provider_cls = cast(
                Callable[..., ComponentProvider], component_provider_cls
            )
            provider = none_throws(
                provider_cls(scheduler_info.name, scheduler_info.image)
            )
            provider.setUp()
            app_def = provider.get_app_def()
            ComponentUtils.run_appdef_on_scheduler(
                app_def, scheduler_info.name, scheduler_info.runconfig, dryrun
            )
            return app_def.name
        finally:
            if provider:
                provider.tearDown()

    def _wait_and_print_report(self, futures: List[AppDefRun]) -> None:
        app_runs: Dict[str, List[str]] = {}
        deadline = time.monotonic() + self._timeout
        for future in futures:
            task_timeout = max(0, int(deadline - time.monotonic()))
            status, msg = self._get_app_def_run_status(future, task_timeout)
            if status not in app_runs:
                app_runs[status] = []
            app_runs[status].append(msg)
        for status, msgs in app_runs.items():
            print(f"\nStatus: `{status}`\n")
            print("\n".join(msgs))
            print("\n")

    def _get_app_def_run_status(
        self, app_def_run: AppDefRun, timeout: int
    ) -> Tuple[str, str]:
        try:
            print(f"Retrieving results from {app_def_run}: {app_def_run.provider_cls}")
            app_name = app_def_run.future.result(timeout=timeout)
            return "succeeded", f"`{app_name}`:`{app_def_run}`"
        except Exception as e:
            stack_trace_msg = traceback.format_exc().replace("\n", "\n  ")

            msg = (
                f"Failure while running: {app_def_run}, provider: `{app_def_run.provider_cls}`: {e}\n"
                f"Stacktrace: {stack_trace_msg}\n"
            )
            return "failed", msg
