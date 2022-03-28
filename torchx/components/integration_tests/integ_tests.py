# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import time
import traceback
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from string import Template
from types import ModuleType
from typing import Callable, List, Mapping, Optional, Tuple, Type, cast

from pyre_extensions import none_throws
from torchx.components.component_test_base import ComponentUtils
from torchx.components.integration_tests.component_provider import ComponentProvider
from torchx.specs import AppDef, CfgVal


@dataclass
class SchedulerInfo:
    name: str
    image: str
    cfg: Mapping[str, CfgVal] = field(default_factory=dict)


@dataclass
class AppDefRun:
    provider: ComponentProvider
    scheduler_info: SchedulerInfo
    app_def: AppDef
    future: Optional[Future] = None


_SUCCESS_APP_FORMAT_TEMPLATE = """
  Name: ${name}
  Scheduler: ${scheduler}"""

_FAIL_APP_FORMAT_TEMPLATE = """
  Name: ${name}
  Provider: ${provider}
  Scheduler: ${scheduler}
  Image: ${image}
  Error: ${error}"""

_REPORT_FORMAT_TEMPLATE = """
${boarder}
Status: Success
${boarder}
${success_report}
\n
${boarder}
Status: Failed
${boarder}
${fail_report}
"""


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
        run_in_parallel: bool = True,
    ) -> None:
        component_providers_cls = self._get_component_providers(module)
        app_runs: List[AppDefRun] = []
        executor = ThreadPoolExecutor()
        for scheduler_info in scheduler_infos:
            sched_futures = self._run_component_providers(
                executor, component_providers_cls, scheduler_info, dryrun
            )
            app_runs += sched_futures
        self._wait_and_print_report(app_runs)

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
        run_in_parallel: bool = True,
    ) -> List[AppDefRun]:
        if run_in_parallel:
            return self._run_component_providers_in_parallel(
                executor, component_providers_cls, scheduler_info, dryrun
            )
        else:
            return self._run_component_providers_in_sequence(
                component_providers_cls, scheduler_info, dryrun
            )

    def _run_component_providers_in_sequence(
        self,
        component_providers_cls: List[Type[ComponentProvider]],
        scheduler_info: SchedulerInfo,
        dryrun: bool = False,
    ) -> List[AppDefRun]:
        app_runs: List[AppDefRun] = []
        for provider_cls in component_providers_cls:
            provider = self._get_app_def_provider(provider_cls, scheduler_info)
            self._run_component_provider(provider, scheduler_info, dryrun)
            app_runs.append(
                AppDefRun(
                    provider=provider,
                    scheduler_info=scheduler_info,
                    future=None,
                    app_def=provider.get_app_def(),
                )
            )
        return app_runs

    def _run_component_providers_in_parallel(
        self,
        executor: ThreadPoolExecutor,
        component_providers_cls: List[Type[ComponentProvider]],
        scheduler_info: SchedulerInfo,
        dryrun: bool = False,
    ) -> List[AppDefRun]:
        app_runs: List[AppDefRun] = []
        for provider_cls in component_providers_cls:
            provider = self._get_app_def_provider(provider_cls, scheduler_info)
            future = executor.submit(
                self._run_component_provider, provider, scheduler_info, dryrun
            )
            app_runs.append(
                AppDefRun(
                    provider=provider,
                    scheduler_info=scheduler_info,
                    future=future,
                    app_def=provider.get_app_def(),
                )
            )
        return app_runs

    def _get_app_def_provider(
        self,
        component_provider_cls: Type[ComponentProvider],
        scheduler_info: SchedulerInfo,
    ) -> ComponentProvider:
        provider_cls = cast(Callable[..., ComponentProvider], component_provider_cls)
        return none_throws(provider_cls(scheduler_info.name, scheduler_info.image))

    def _run_component_provider(
        self,
        provider: ComponentProvider,
        scheduler_info: SchedulerInfo,
        dryrun: bool = False,
    ) -> None:
        try:
            provider.setUp()
            ComponentUtils.run_appdef_on_scheduler(
                provider.get_app_def(),
                scheduler_info.name,
                scheduler_info.cfg,
                dryrun,
            )
        finally:
            provider.tearDown()

    def _wait_and_print_report(self, app_runs: List[AppDefRun]) -> None:
        succeeded_apps: List[AppDefRun] = []
        failed_apps: List[Tuple[AppDefRun, str]] = []
        deadline = time.monotonic() + self._timeout
        for app_run in app_runs:
            task_timeout = max(0, int(deadline - time.monotonic()))
            error_msg = self._get_app_def_run_status(app_run, task_timeout)
            if not error_msg:
                succeeded_apps.append(app_run)
            else:
                failed_apps.append((app_run, error_msg))
        success_report = ""
        for app_run in succeeded_apps:
            success_report_run = Template(_SUCCESS_APP_FORMAT_TEMPLATE).substitute(
                name=app_run.app_def.name, scheduler=app_run.scheduler_info.name
            )
            success_report += f"{success_report_run}\n"
        fail_report = ""
        for app_run, error_msg in failed_apps:
            fail_report_run = Template(_FAIL_APP_FORMAT_TEMPLATE).substitute(
                name=app_run.app_def.name,
                provider=app_run.provider,
                scheduler=app_run.scheduler_info.name,
                image=app_run.scheduler_info.image,
                error=error_msg,
            )
            fail_report += f"{fail_report_run}\n"
        delim = "*"
        width = 80
        msg = Template(_REPORT_FORMAT_TEMPLATE).substitute(
            boarder=delim * width,
            success_report=success_report or "<NONE>",
            fail_report=fail_report or "<NONE>",
        )
        print(msg)
        if len(failed_apps) > 0:
            raise RuntimeError(
                "Component test failed, see report above for detailed issue"
            )

    def _get_app_def_run_status(
        self, app_def_run: AppDefRun, timeout: int
    ) -> Optional[str]:
        try:
            print(f"Retrieving {app_def_run.app_def.name}: {app_def_run.provider}")
            if app_def_run.future:
                app_def_run.future.result(timeout=timeout)
            return None
        except Exception:
            stack_trace_msg = traceback.format_exc().replace("\n", "\n  ")
            return stack_trace_msg
