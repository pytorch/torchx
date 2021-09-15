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
from typing import Callable, List, Any, Dict, Optional

from torchx.runner import get_runner
from torchx.specs import AppDef, SchedulerBackend, RunConfig, AppState, AppHandle
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

    def _validate(self, module: ModuleType, function_name: str) -> None:
        """
        _validate takes in a module and the name of a component function
        definition defined in that module and validates that it is a valid
        component def.
        """
        file_path = path = os.path.abspath(module.__file__)
        with open(file_path, "r") as fp:
            file_content = fp.read()
        linter_errors = validate(file_content, file_path, function_name)
        if len(linter_errors) != 0:
            error_msg = ""
            for linter_error in linter_errors:
                error_msg += f"Lint Error: {linter_error.description}\n"
            raise ValueError(error_msg)
        self.assertEquals(0, len(linter_errors))

    def _validate_component(self, component: Callable[..., AppDef]) -> None:
        module_name = component.__module__
        component_name = component.__name__
        module = sys.modules[module_name]
        filepath = module.__file__
        with open(filepath, "r") as fp:
            source = fp.read()
        errors = validate(source=source, torchx_function=component_name)
        if len(errors) != 0:
            raise ValueError(
                f"Component {component_name} has the following linter errors: {errors}"
            )

    def _run_component_on_scheduler(
        self,
        component: Callable[..., AppDef],
        component_args: List[Any],
        component_kwargs: Dict[str, Any],
        scheduler: SchedulerBackend,
        scheduler_cfg: RunConfig,
    ) -> None:
        app_def = component(*component_args, **component_kwargs)
        runner = get_runner("test-runner")
        app_handle = runner.run(app_def, scheduler, scheduler_cfg)
        print(f"AppHandle: {app_handle}")
        app_status = runner.wait(app_handle)
        print(f"Final status: {app_status}")
        assert app_status.state == AppState.SUCCEEDED
        return app_handle

    def run_component_on_local(
        self,
        component: Callable[..., AppDef],
        component_args: Optional[List[Any]] = None,
        component_kwargs: Optional[Dict[str, Any]] = None,
    ) -> AppHandle:
        component_args = component_args or []
        component_kwargs = component_kwargs or {}
        self._validate_component(component)
        app_def = component(*component_args, **component_kwargs)
        cfg = RunConfig()
        cfg.set("img_type", "dir")
        self._run_component_on_scheduler(
            component, component_args, component_kwargs, "local", cfg
        )
