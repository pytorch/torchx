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
import unittest
from types import ModuleType
from typing import Mapping, Union

from pyre_extensions import none_throws
from torchx.runner import get_runner
from torchx.specs import (
    AppDef,
    AppDryRunInfo,
    AppHandle,
    AppState,
    CfgVal,
    SchedulerBackend,
)
from torchx.specs.api import _create_args_parser
from torchx.specs.finder import get_component


class ComponentUtils:
    @classmethod
    def run_appdef_on_scheduler(
        cls,
        app_def: AppDef,
        scheduler: SchedulerBackend,
        cfg: Mapping[str, CfgVal],
        dryrun: bool = False,
    ) -> Union[AppHandle, AppDryRunInfo]:
        """
        Runs component on provided scheduler.
        """

        runner = get_runner("test-runner")
        if dryrun:
            dryrun_info = runner.dryrun(app_def, scheduler, cfg)
            print(f"Dryrun info: {dryrun_info}")
            return dryrun_info
        else:
            app_handle = runner.run(app_def, scheduler, cfg)
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

    .. doctest:: [component_test_case]

     import unittest
     from torchx.components.component_test_base import ComponentTestCase
     from torchx.components import utils
     class MyComponentTest(ComponentTestCase):
         def test_my_comp(self):
             self.validate(utils, "copy")
     MyComponentTest("test_my_comp").run()

    """

    def validate(self, module: ModuleType, function_name: str) -> None:
        """
        Validates the component by effectively running:

        .. code-block:: shell-session

         $ torchx run COMPONENT.py:FN --help

        """

        # make it the same as a custom component (e.g. /abs/path/to/component.py:train)
        component_id = f"{os.path.abspath(module.__file__)}:{function_name}"
        component_def = get_component(component_id)

        # on `--help` argparse will print the help message and exit 0
        # just make sure that happens; if the component has errors then
        # this will raise an exception and the test will fail
        with self.assertRaises(SystemExit):
            _ = _create_args_parser(component_def.fn).parse_args(["--help"])
