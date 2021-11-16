#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import io
import os
import shutil
import signal
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List
from unittest.mock import MagicMock, patch

from torchx.cli.cmd_run import CmdBuiltins, CmdRun, _parse_run_config, logger
from torchx.schedulers.local_scheduler import SignalException
from torchx.specs import runopts


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
        shutil.rmtree(self.tmpdir)

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
                "--scheduler",
                "local_cwd",
                "utils.echo",
                "--image",
                "/tmp",
            ]
        )
        self.cmd_run.run(args)
        mock_runner_run.assert_not_called()

    def test_runopts_not_found(self) -> None:
        args = self.parser.parse_args(
            [
                "--dryrun",
                "--scheduler",
                "kubernetes",
                "utils.echo",
                "--image",
                "/tmp",
            ]
        )
        with patch.object(logger, "error") as log_error:
            with self.assertRaises(SystemExit) as e:
                self.cmd_run.run(args)
            msg = log_error.call_args[0][0]
            self.assertTrue(
                "Scheduler arg is incorrect or missing required option" in msg
            )

    def _get_test_runopts(self) -> runopts:
        opts = runopts()
        opts.add("foo", type_=str, default="", help="")
        opts.add("test_key", type_=str, default="", help="")
        opts.add("default_time", type_=int, default=0, help="")
        opts.add("enable", type_=bool, default=True, help="")
        opts.add("disable", type_=bool, default=True, help="")
        opts.add("complex_list", type_=List[str], default=[], help="")
        return opts

    def test_parse_runopts(self) -> None:
        test_arg = "foo=bar,test_key=test_value,default_time=42,enable=True,disable=False,complex_list=v1;v2;v3"
        expected_args = {
            "foo": "bar",
            "test_key": "test_value",
            "default_time": 42,
            "enable": True,
            "disable": False,
            "complex_list": ["v1", "v2", "v3"],
        }
        opts = self._get_test_runopts()
        runconfig = _parse_run_config(test_arg, opts)
        for k, v in expected_args.items():
            self.assertEqual(v, runconfig.get(k))


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
