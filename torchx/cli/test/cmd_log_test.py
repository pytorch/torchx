#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import unittest
from typing import Optional, Iterator
from unittest.mock import patch, MagicMock

from torchx.cli.cmd_log import ENDC, GREEN, get_logs
from torchx.specs.api import Application, Role, parse_app_handle


class SentinelError(Exception):
    """
    Used to mock sys.exit
    """

    pass


SESSION = "torchx.specs.lib.run"


class MockSession:
    def __call__(self, name: Optional[str] = None) -> "MockSession":
        return self

    def describe(self, app_handle: str) -> Application:
        scheduler_backend, session_name, app_id = parse_app_handle(app_handle)
        return Application(name=app_id).of(
            Role(name="master").replicas(1), Role(name="trainer").replicas(3)
        )

    def log_lines(
        self,
        app_id: str,
        role_name: str,
        k: str,
        regex: str,
        since: Optional[int] = None,
        until: Optional[int] = None,
        should_tail: bool = False,
    ) -> Iterator[str]:
        import re

        if regex is None:
            regex = ".*"

        log_lines = ["INFO foo", "ERROR bar", "WARN baz"]
        return iter([line for line in log_lines if re.match(regex, line)])


class CmdLogTest(unittest.TestCase):
    @patch("sys.exit", side_effect=SentinelError)
    def test_cmd_log_bad_job_identifier(self, exit_mock: MagicMock) -> None:
        with self.assertRaises(SentinelError):
            get_logs("local:///SparseNNApplication/", "QPS.*")
        exit_mock.assert_called_once_with(1)

    @patch(SESSION, new_callable=MockSession)
    @patch("sys.exit", side_effect=SentinelError)
    def test_cmd_log_unknown_role(
        self, exit_mock: MagicMock, session_mock: MagicMock
    ) -> None:
        with self.assertRaises(SentinelError):
            get_logs(
                "local://default/SparseNNApplication/unknown_role",
                regex=None,
            )

        exit_mock.assert_called_once_with(1)

    @patch(SESSION, new_callable=MockSession)
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_cmd_log_all_replicas(
        self, stdout_mock: MagicMock, session_mock: MagicMock
    ) -> None:
        get_logs("local://test-session/SparseNNApplication/trainer", regex="INFO")
        self.assertSetEqual(
            {
                f"{GREEN}trainer/0{ENDC} INFO foo",
                f"{GREEN}trainer/1{ENDC} INFO foo",
                f"{GREEN}trainer/2{ENDC} INFO foo",
                # print writes the final newline so we
                # end up with an empty string when we split by \n
                "",
            },
            set(stdout_mock.getvalue().split("\n")),
        )

    @patch(SESSION, new_callable=MockSession)
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_cmd_log_one_replica(
        self, stdout_mock: MagicMock, session_mock: MagicMock
    ) -> None:
        get_logs("local://test-session/SparseNNApplication/trainer/0", regex=None)
        self.assertSetEqual(
            {
                f"{GREEN}trainer/0{ENDC} INFO foo",
                f"{GREEN}trainer/0{ENDC} ERROR bar",
                f"{GREEN}trainer/0{ENDC} WARN baz",
                # print writes the final newline so we
                # end up with an empty string when we split by \n
                "",
            },
            set(stdout_mock.getvalue().split("\n")),
        )

    @patch(SESSION, new_callable=MockSession)
    @patch("sys.stdout", new_callable=io.StringIO)
    def test_cmd_log_some_replicas(
        self, stdout_mock: MagicMock, session_mock: MagicMock
    ) -> None:
        get_logs("local://test-session/SparseNNApplication/trainer/0,2", regex="WARN")
        self.assertSetEqual(
            {
                f"{GREEN}trainer/0{ENDC} WARN baz",
                f"{GREEN}trainer/2{ENDC} WARN baz",
                # print writes the final newline so we
                # end up with an empty string when we split by \n
                "",
            },
            set(stdout_mock.getvalue().split("\n")),
        )

    @patch(SESSION, new_callable=MockSession)
    def test_print_log_lines_throws(self, session_mock: MagicMock) -> None:
        # makes sure that when the function executed in the threadpool
        # errors out; we raise the exception all the way through
        with patch.object(session_mock, "log_lines") as log_lines_mock:
            log_lines_mock.side_effect = RuntimeError
            with self.assertRaises(RuntimeError):
                get_logs(
                    "local://test-session/SparseNNApplication/trainer/0,1", regex=None
                )
