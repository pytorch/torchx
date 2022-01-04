# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Mapping
from unittest.mock import MagicMock, call

from torchx.runner.workspaces import (
    get_workspace_runner,
    WorkspaceRunner,
)
from torchx.schedulers.api import (
    WorkspaceScheduler,
)
from torchx.specs.api import AppDryRunInfo, AppDef, CfgVal


class WorkspaceRunnerTest(unittest.TestCase):
    def test_get_workspace_runner(self) -> None:
        self.assertIsInstance(get_workspace_runner(), WorkspaceRunner)

    def test_workspace_runner(self) -> None:
        scheduler = MagicMock(spec=WorkspaceScheduler)

        def submit_dryrun(app: AppDef, cfg: Mapping[str, CfgVal]) -> AppDryRunInfo[str]:
            self.assertEqual(app.roles[0].image, "$img")

            dryrun_info: AppDryRunInfo[str] = AppDryRunInfo(str("req"), str)
            dryrun_info._app = app
            return dryrun_info

        scheduler.submit_dryrun = submit_dryrun
        scheduler.build_workspace_image.return_value = "$img"
        runner = WorkspaceRunner(
            "workspaces_test",
            schedulers={
                "mock": scheduler,
            },
        )
        app_args = ["--image", "dummy_image", "--script", "test.py"]
        workspace = "memory:///foo"
        ret = runner.run_component("dist.ddp", app_args, "mock", workspace)

        self.assertEqual(
            scheduler.build_workspace_image.mock_calls,
            [
                call("dummy_image", workspace),
            ],
        )
