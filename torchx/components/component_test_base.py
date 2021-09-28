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
from typing import Callable, Union, List

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
from torchx.specs.file_linter import validate, LinterMessage


class ComponentUtils:
    @classmethod
    def get_linter_errors(
        cls, file_path: str, function_name: str
    ) -> List[LinterMessage]:
        with open(file_path, "r") as fp:
            file_content = fp.read()
        linter_errors = validate(file_content, file_path, function_name)
        if len(linter_errors) != 0:
            error_msg = ""
            for linter_error in linter_errors:
                error_msg += f"Lint Error: {linter_error.description}\n"
            raise ValueError(error_msg)
        return linter_errors

    @classmethod
    def get_linter_errors_for_component(
        cls, component: Callable[..., AppDef]
    ) -> List[LinterMessage]:
        """
        _validate_component takes in a function that produces a component
        and validates that it is a valid component def.
        """
        module_name = component.__module__
        function_name = component.__name__
        module = sys.modules[module_name]
        file_path = module.__file__
        return cls.get_linter_errors(file_path, function_name)

    @classmethod
    def run_appdef_on_scheduler(
        cls,
        app_def: AppDef,
        scheduler: SchedulerBackend,
        scheduler_cfg: RunConfig,
        dryrun: bool = False,
    ) -> Union[AppHandle, AppDryRunInfo]:
        """
        Runs component on provided scheduler.
        """

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
        file_path = os.path.abspath(module.__file__)
        linter_errors = ComponentUtils.get_linter_errors(file_path, function_name)
        self.assertListEqual([], linter_errors)
