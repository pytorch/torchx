# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
You can unit test the component definitions as you would normal Python code
since they are valid Python definitions.

We do recommend using :py:class:`ComponentTestCase` to ensure that your
component can be parsed by the TorchX CLI. The CLI requires stricter formatting
on the doc string than pure Python as the doc string is used for parsing CLI
args.
"""

import os
import sys
import unittest
from types import ModuleType
<<<<<<< HEAD
from typing import Callable, Union
=======
from typing import Callable, List, Any, Dict, Optional, Union
>>>>>>> b1e1ade8e2f5a6ea5e244e47f65a4b0b104d49f9

from pyre_extensions import none_throws
from torchx.runner import get_runner
from torchx.specs import (
    AppDef,
    SchedulerBackend,
    RunConfig,
    AppState,
    AppHandle,
    AppDryRunInfo,
)
from torchx.specs.file_linter import validate


class ComponentTestCase(unittest.TestCase):
    """
    ComponentTestCase is an extension of TestCase with helper methods for use
    with testing component definitions.

    >>> import unittest
    >>> from torchx.components.component_test_base import ComponentTestCase
    >>> from torchx.components import utils
    >>> class MyComponentTest(ComponentTestCase):
    ...     def test_my_comp(self):
    ...         self._validate(utils, "copy")
    >>> MyComponentTest("test_my_comp").run()
    <unittest.result.TestResult run=1 errors=0 failures=0>
    """

    def _validate_content(self, file_path: str, function_name: str) -> None:
        with open(file_path, "r") as fp:
            file_content = fp.read()
        linter_errors = validate(file_content, file_path, function_name)
        if len(linter_errors) != 0:
            error_msg = ""
            for linter_error in linter_errors:
                error_msg += f"Lint Error: {linter_error.description}\n"
            raise ValueError(error_msg)
        self.assertEquals(0, len(linter_errors))

    def _validate(self, module: ModuleType, function_name: str) -> None:
        """
        _validate takes in a module and the name of a component function
        definition defined in that module and validates that it is a valid
        component def.
        """
        file_path = os.path.abspath(module.__file__)
        self._validate_content(file_path, function_name)

    def _validate_component(self, component: Callable[..., AppDef]) -> None:
        """
        _validate_component takes in a function that produces a component
        and validates that it is a valid component def.
        """
        module_name = component.__module__
        function_name = component.__name__
        module = sys.modules[module_name]
        file_path = module.__file__
        self._validate_content(file_path, function_name)

<<<<<<< HEAD
    def run_appdef_on_scheduler(
        self,
        app_def: AppDef,
=======
    def _run_component_on_scheduler(
        self,
        component: Callable[..., AppDef],
        # pyre-ignore[2]
        component_args: List[Any],
        component_kwargs: Dict[str, Any],
>>>>>>> b1e1ade8e2f5a6ea5e244e47f65a4b0b104d49f9
        scheduler: SchedulerBackend,
        scheduler_cfg: RunConfig,
        dryrun: bool = False,
    ) -> Union[AppHandle, AppDryRunInfo]:
<<<<<<< HEAD
        """
        Runs component on provided scheduler.
        """

=======
        app_def = component(*component_args, **component_kwargs)
>>>>>>> b1e1ade8e2f5a6ea5e244e47f65a4b0b104d49f9
        runner = get_runner("test-runner")
        if dryrun:
            dryrun_info = runner.dryrun(app_def, scheduler, scheduler_cfg)
            print(f"Dryrun info: {dryrun_info}")
            return dryrun_info
        else:
            app_handle = runner.run(app_def, scheduler, scheduler_cfg)
            print(f"AppHandle: {app_handle}")
            app_status = runner.wait(app_handle)
            print(f"Final status: {app_status}")
            if none_throws(app_status).state != AppState.SUCCEEDED:
                raise AssertionError(
                    f"App {app_handle} failed with status: {app_status}"
                )
            return app_handle
<<<<<<< HEAD
=======

    def run_component_on_local(
        self,
        component: Callable[..., AppDef],
        # pyre-ignore[2]
        component_args: Optional[List[Any]] = None,
        component_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Union[AppHandle, AppDryRunInfo]:
        component_args = component_args or []
        component_kwargs = component_kwargs or {}
        self._validate_component(component)
        cfg = RunConfig()
        cfg.set("img_type", "dir")
        return self._run_component_on_scheduler(
            component, component_args, component_kwargs, "local", cfg
        )

    def run_component_on_k8s(
        self,
        component: Callable[..., AppDef],
        # pyre-ignore[2]
        component_args: Optional[List[Any]] = None,
        component_kwargs: Optional[Dict[str, Any]] = None,
        dryrun: bool = False,
    ) -> Union[AppHandle, AppDryRunInfo]:
        component_args = component_args or []
        component_kwargs = component_kwargs or {}
        self._validate_component(component)
        cfg = RunConfig()
        cfg.set("namespace", "torchx-dev")
        cfg.set("queue", "default")
        return self._run_component_on_scheduler(
            component, component_args, component_kwargs, "kubernetes", cfg, dryrun
        )
>>>>>>> b1e1ade8e2f5a6ea5e244e47f65a4b0b104d49f9
