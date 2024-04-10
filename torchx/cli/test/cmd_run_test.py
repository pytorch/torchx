#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import dataclasses
import io
import os
import shutil
import signal
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

from torchx.cli.argparse_util import ArgOnceAction, torchxconfig

from torchx.cli.cmd_run import _parse_component_name_and_args, CmdBuiltins, CmdRun
from torchx.schedulers.local_scheduler import SignalException

from torchx.specs import AppDryRunInfo


@contextmanager
def cwd(path: str) -> Generator[None, None, None]:
    orig_cwd = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(orig_cwd)


class CmdRunTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = Path(tempfile.mkdtemp())
        self.parser = argparse.ArgumentParser()
        self.cmd_run = CmdRun()
        self.cmd_run.add_arguments(self.parser)

    def tearDown(self) -> None:
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        ArgOnceAction.called_args = set()
        torchxconfig.called_args = set()

    def test_run_with_multiple_scheduler_args(self) -> None:

        args = ["--scheduler_args", "first_args", "--scheduler_args", "second_args"]
        with self.assertRaises(SystemExit) as cm:
            self.parser.parse_args(args)
        self.assertEqual(cm.exception.code, 1)

    def test_run_with_multiple_schedule_args(self) -> None:

        args = [
            "--scheduler",
            "local_cwd",
            "--scheduler",
            "local_cwd",
        ]

        with self.assertRaises(SystemExit) as cm:
            self.parser.parse_args(args)
        self.assertEqual(cm.exception.code, 1)

    def test_run_with_user_conf_abs_path(self) -> None:
        args = self.parser.parse_args(
            [
                "--scheduler",
                "local_cwd",
                str(Path(__file__).parent / "components.py:touch"),
                "--file",
                str(self.tmpdir / "foobar.txt"),
            ]
        )
        self.cmd_run.run(args)
        self.assertEqual("cli_run", os.environ["TORCHX_CONTEXT_NAME"])
        self.assertTrue(os.path.isfile(str(self.tmpdir / "foobar.txt.test")))

    def test_run_with_relpath(self) -> None:
        # should pick up test/examples/touch.torchx (not the builtin)
        with cwd(str(Path(__file__).parent)):
            args = self.parser.parse_args(
                [
                    "--scheduler",
                    "local_cwd",
                    str(Path(__file__).parent / "components.py:touch_v2"),
                    "--file",
                    str(self.tmpdir / "foobar.txt"),
                ]
            )

            self.cmd_run.run(args)
            self.assertTrue(os.path.isfile(str(self.tmpdir / "foobar.txt.testv2")))

    @patch("sys.stderr", new_callable=io.StringIO)
    def test_run_with_log(self, stderr: io.StringIO) -> None:
        with cwd(str(Path(__file__).parent)):
            args = self.parser.parse_args(
                [
                    "--log",
                    "--wait",
                    "--scheduler",
                    "local_cwd",
                    "-cfg",
                    f"log_dir={self.tmpdir}",
                    str(Path(__file__).parent / "components.py:echo_stderr"),
                    "--msg",
                    "toast",
                ]
            )

            self.cmd_run.run(args)
        self.assertRegex(stderr.getvalue(), "echo/0.*toast")

    @patch(
        "torchx.runner.Runner.wait", side_effect=SignalException("msg", signal.SIGTERM)
    )
    @patch("torchx.schedulers.local_scheduler.LocalScheduler.close")
    def test_run_terminate_on_received_signal(
        self,
        mock_scheduler_close: MagicMock,
        _,
    ) -> None:
        with cwd(str(Path(__file__).parent)):
            args = self.parser.parse_args(
                [
                    "--scheduler",
                    "local_cwd",
                    str(Path(__file__).parent / "components.py:touch_v2"),
                    "--file",
                    str(self.tmpdir / "foobar.txt"),
                ]
            )

            with self.assertRaises(SignalException):
                self.cmd_run.run(args)

            mock_scheduler_close.assert_called()

    def test_run_missing(self) -> None:
        args = self.parser.parse_args(
            [
                "--scheduler",
                "local_cwd",
                "1234_does_not_exist.torchx",
            ]
        )

        with self.assertRaises(SystemExit):
            self.cmd_run.run(args)

    def test_conf_file_missing(self) -> None:
        # guards against existing .torchxconfig files
        # in user's $HOME or the CWD where the test is launched from
        with patch(
            "torchx.runner.config.DEFAULT_CONFIG_DIRS", return_value=[self.tmpdir]
        ):
            with self.assertRaises(SystemExit):
                args = self.parser.parse_args(
                    [
                        "--scheduler",
                        "local_cwd",
                    ]
                )
                self.cmd_run.run(args)

    @patch("torchx.runner.Runner.run")
    def test_run_dryrun(self, mock_runner_run: MagicMock) -> None:
        args = self.parser.parse_args(
            [
                "--dryrun",
                "--parent_run_id",
                "experiment_1",
                "--scheduler",
                "local_cwd",
                "utils.echo",
                "--image",
                "/tmp",
            ]
        )
        self.cmd_run.run(args)
        mock_runner_run.assert_not_called()

    @patch("torchx.runner.Runner.dryrun_component")
    def test_store_experiment_id(self, mock_runner_run: MagicMock) -> None:
        args = self.parser.parse_args(
            [
                "--dryrun",
                "--parent_run_id",
                "experiment_1",
                "--scheduler",
                "local_cwd",
                "utils.echo",
                "--image",
                "/tmp",
            ]
        )

        app_run_info_stub = AppDryRunInfo("req", lambda x: x)
        req_type_dataclass = dataclasses.make_dataclass("T", [])
        app_run_info_stub._app = req_type_dataclass()
        mock_runner_run.return_value = app_run_info_stub

        self.cmd_run.run(args)

        # compatible with python 3.7
        call_kwargs = mock_runner_run.call_args[-1]
        self.assertEqual(call_kwargs["parent_run_id"], "experiment_1")

    def test_parse_component_name_and_args_no_default(self) -> None:
        sp = argparse.ArgumentParser(prog="test")
        self.assertEqual(
            ("utils.echo", []),
            _parse_component_name_and_args(["utils.echo"], sp),
        )
        self.assertEqual(
            ("utils.echo", []),
            _parse_component_name_and_args(["--", "utils.echo"], sp),
        )
        self.assertEqual(
            ("utils.echo", ["--msg", "hello"]),
            _parse_component_name_and_args(["utils.echo", "--msg", "hello"], sp),
        )

        with self.assertRaises(SystemExit):
            _parse_component_name_and_args(["--"], sp)

        with self.assertRaises(SystemExit):
            _parse_component_name_and_args(["--msg", "hello"], sp)

        with self.assertRaises(SystemExit):
            _parse_component_name_and_args(["-m", "hello"], sp)

    def test_parse_component_name_and_args_with_default(self) -> None:
        sp = argparse.ArgumentParser(prog="test")
        dirs = [str(self.tmpdir)]

        with open(Path(self.tmpdir) / ".torchxconfig", "w") as f:
            f.write(
                """#
[cli:run]
component = custom.echo
            """
            )

        self.assertEqual(
            ("utils.echo", []), _parse_component_name_and_args(["utils.echo"], sp, dirs)
        )
        self.assertEqual(
            ("utils.echo", ["--msg", "hello"]),
            _parse_component_name_and_args(["utils.echo", "--msg", "hello"], sp, dirs),
        )
        self.assertEqual(
            ("custom.echo", []),
            _parse_component_name_and_args([], sp, dirs),
        )
        self.assertEqual(
            ("custom.echo", []),
            _parse_component_name_and_args(["--"], sp, dirs),
        )
        self.assertEqual(
            ("custom.echo", ["--msg", "hello"]),
            _parse_component_name_and_args(["--", "--msg", "hello"], sp, dirs),
        )
        self.assertEqual(
            ("custom.echo", ["--msg", "hello"]),
            _parse_component_name_and_args(["--msg", "hello"], sp, dirs),
        )
        self.assertEqual(
            ("custom.echo", ["-m", "hello"]),
            _parse_component_name_and_args(["-m", "hello"], sp, dirs),
        )


class CmdBuiltinTest(unittest.TestCase):
    def test_run(self) -> None:
        # nothing to assert, just make sure the supported schedulers print runopts

        parser = argparse.ArgumentParser()
        cmd_builtins = CmdBuiltins()
        cmd_builtins.add_arguments(parser)
        args = parser.parse_args([])
        cmd_builtins.run(args)

    def test_builtins(self) -> None:
        cmd_builtins = CmdBuiltins()
        builtins = cmd_builtins._builtins()
        # make sure there's at least one
        # there will always be one (example.torchx)
        self.assertTrue(len(builtins) > 0)
        for component in builtins.values():
            self.assertListEqual([], component.validation_errors)

    def test_print_builtin(self) -> None:
        parser = argparse.ArgumentParser()

        cmd_builtins = CmdBuiltins()
        cmd_builtins.add_arguments(parser)

        cmd_builtins.run(parser.parse_args(["--print", "dist.ddp"]))
        # nothing to assert, just make sure it runs
