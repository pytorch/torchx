#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from datetime import datetime
from typing import Iterable, Mapping, Optional, Union
from unittest.mock import MagicMock, patch

from torchx.schedulers.api import DescribeAppResponse, Scheduler, Stream
from torchx.specs.api import (
    NULL_RESOURCE,
    AppDef,
    AppDryRunInfo,
    CfgVal,
    InvalidRunConfigException,
    Resource,
    runopts,
)


class SchedulerTest(unittest.TestCase):
    class MockScheduler(Scheduler):
        def __init__(self, session_name: str) -> None:
            super().__init__("mock", session_name)

        def schedule(self, dryrun_info: AppDryRunInfo[None]) -> str:
            app = dryrun_info._app
            assert app is not None
            return app.name

        def _submit_dryrun(
            self,
            app: AppDef,
            cfg: Mapping[str, CfgVal],
        ) -> AppDryRunInfo[None]:
            return AppDryRunInfo(None, lambda t: "None")

        def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
            return None

        def _cancel_existing(self, app_id: str) -> None:
            pass

        def log_iter(
            self,
            app_id: str,
            role_name: str,
            k: int = 0,
            regex: Optional[str] = None,
            since: Optional[datetime] = None,
            until: Optional[datetime] = None,
            should_tail: bool = False,
            streams: Optional[Stream] = None,
        ) -> Iterable[str]:
            return iter([])

        def run_opts(self) -> runopts:
            opts = runopts()
            opts.add("foo", type_=str, required=True, help="required option")
            return opts

        def resolve_resource(self, resource: Union[str, Resource]) -> Resource:
            return NULL_RESOURCE

    def test_invalid_run_cfg(self) -> None:
        scheduler_mock = SchedulerTest.MockScheduler("test_session")
        app_mock = MagicMock()

        with self.assertRaises(InvalidRunConfigException):
            empty_cfg = {}
            scheduler_mock.submit(app_mock, empty_cfg)

        with self.assertRaises(InvalidRunConfigException):
            bad_type_cfg = {"foo": 100}
            scheduler_mock.submit(app_mock, bad_type_cfg)

    def test_invalid_dryrun_cfg(self) -> None:
        scheduler_mock = SchedulerTest.MockScheduler("test_session")
        app_mock = MagicMock()

        with self.assertRaises(InvalidRunConfigException):
            empty_cfg = {}
            scheduler_mock.submit_dryrun(app_mock, empty_cfg)

        with self.assertRaises(InvalidRunConfigException):
            bad_type_cfg = {"foo": 100}
            scheduler_mock.submit_dryrun(app_mock, bad_type_cfg)

    def test_role_preproc_called(self) -> None:
        scheduler_mock = SchedulerTest.MockScheduler("test_session")
        app_mock = MagicMock()
        app_mock.roles = [MagicMock()]

        cfg = {"foo": "bar"}
        scheduler_mock.submit_dryrun(app_mock, cfg)
        role_mock = app_mock.roles[0]
        role_mock.pre_proc.assert_called_once()

    def test_validate(self) -> None:
        scheduler_mock = SchedulerTest.MockScheduler("test_session")
        app_mock = MagicMock()
        app_mock.roles = [MagicMock()]
        app_mock.roles[0].resource = NULL_RESOURCE

        with self.assertRaises(ValueError):
            scheduler_mock._validate(app_mock, "local")

    def test_cancel_not_exists(self) -> None:
        scheduler_mock = SchedulerTest.MockScheduler("test_session")
        with patch.object(scheduler_mock, "_cancel_existing") as cancel_mock:
            with patch.object(scheduler_mock, "exists") as exists_mock:
                exists_mock.return_value = True
                scheduler_mock.cancel("test_id")
                cancel_mock.assert_called_once()

    def test_cancel_exists(self) -> None:
        scheduler_mock = SchedulerTest.MockScheduler("test_session")
        with patch.object(scheduler_mock, "_cancel_existing") as cancel_mock:
            with patch.object(scheduler_mock, "exists") as exists_mock:
                exists_mock.return_value = False
                scheduler_mock.cancel("test_id")
                cancel_mock.assert_not_called()

    def test_close_twice(self) -> None:
        scheduler_mock = SchedulerTest.MockScheduler("test")
        scheduler_mock.close()
        scheduler_mock.close()
        # nothing to validate explicitly, just that no errors are raised
