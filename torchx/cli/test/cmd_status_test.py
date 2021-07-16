#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import time
import unittest
from unittest.mock import patch

from torchx.cli.cmd_status import CmdStatus, format_app_status, format_error_message
from torchx.specs.api import AppState, AppStatus, ReplicaStatus, RoleStatus


class CmdStatusTest(unittest.TestCase):
    def test_run(self) -> None:
        parser = argparse.ArgumentParser()
        cmd_status = CmdStatus()
        cmd_status.add_arguments(parser)
        args = parser.parse_args(["local://test_session/test_app"])

        for app_status in [None, AppStatus(state=AppState.RUNNING)]:
            with self.subTest(app_status=app_status):
                with patch("torchx.runner.api.Runner.status") as status_mock:
                    status_mock.return_value = app_status

                    try:
                        cmd_status.run(args)
                        exit_code = None
                    except SystemExit as e:
                        exit_code = e.code

                    status_mock.assert_called_once_with(args.app_handle)

                    if app_status is None:
                        self.assertEqual(exit_code, 1)
                    else:
                        self.assertIsNone(exit_code)

    def test_format_error_message(self) -> None:
        rpc_error_message = """RuntimeError('On WorkerInfo(id=1, name=trainer:0:0):
RuntimeError(ShardingError('Table of size 715.26GB cannot be added to any rank'))
Traceback (most recent call last):
..
')
Traceback (most recent call last):
  File "/dev/shm/uid-0/360e3568-seed-nspid4026541870-ns-4026541866/torch/distributed/rpc/internal.py", line 190, in _run_function
"""
        expected_error_message = """RuntimeError('On WorkerInfo(id=1, name=trainer:0:0):
RuntimeError(ShardingError('Table
 of size 715.26GB cannot be added to any rank'))
Traceback (most recent call last):
..
')"""
        actual_message = format_error_message(rpc_error_message, header="", width=80)
        self.assertEqual(expected_error_message, actual_message)

    def _get_test_app_status(self) -> AppStatus:
        error_msg = '{"message":{"message":"error","errorCode":-1,"extraInfo":{"timestamp":1293182}}}'
        replica1 = ReplicaStatus(
            id=0,
            state=AppState.FAILED,
            role="worker",
            hostname="localhost",
            structured_error_msg=error_msg,
        )

        replica2 = ReplicaStatus(
            id=1,
            state=AppState.RUNNING,
            role="worker",
            hostname="localhost",
        )

        role_status = RoleStatus(role="worker", replicas=[replica1, replica2])
        return AppStatus(state=AppState.RUNNING, roles=[role_status])

    def test_format_app_status(self) -> None:
        os.environ["TZ"] = "Europe/London"
        time.tzset()

        app_status = self._get_test_app_status()
        actual_message = format_app_status(app_status)
        print(actual_message)
        expected_message = """AppDef:
  State: RUNNING
  Num Restarts: 0
Roles:
 *worker[0]:FAILED (exitcode: -1)
    timestamp: 1970-01-16 00:13:02
    hostname: localhost
    error_msg: error
  worker[1]:RUNNING"""
        self.assertEqual(expected_message, actual_message)
