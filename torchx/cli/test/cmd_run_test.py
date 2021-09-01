#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import shutil
import signal
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from typing import Generator
from unittest.mock import MagicMock, patch

from torchx.cli.cmd_run import CmdBuiltins, CmdRun, _parse_run_config
from torchx.schedulers.local_scheduler import SignalException


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
                "local",
                str(Path(__file__).parent / "components.py:touch"),
                "--file",
                str(self.tmpdir / "foobar.txt"),
            ]
        )
        self.cmd_run.run(args)
        self.assertTrue(os.path.isfile(str(self.tmpdir / "foobar.txt.test")))

    def test_run_with_relpath(self) -> None:
        # should pick up test/examples/touch.torchx (not the builtin)
        with cwd(str(Path(__file__).parent)):
            args = self.parser.parse_args(
                [
                    "--scheduler",
                    "local",
                    str(Path(__file__).parent / "components.py:touch_v2"),
                    "--file",
                    str(self.tmpdir / "foobar.txt"),
                ]
            )

            self.cmd_run.run(args)
            self.assertTrue(os.path.isfile(str(self.tmpdir / "foobar.txt.testv2")))

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
                    "local",
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
                "local",
                "1234_does_not_exist.torchx",
            ]
        )
        self.cmd_run.run(args)

    @patch("torchx.runner.Runner.run")
    def test_run_dryrun(self, mock_runner_run: MagicMock) -> None:
        args = self.parser.parse_args(
            [
                "--dryrun",
                "--scheduler",
                "local",
                "utils.echo",
            ]
        )
        self.cmd_run.run(args)
        mock_runner_run.assert_not_called()

    def test_parse_run_config(self) -> None:
        args = "key=value,foo=bar"
        cfg = _parse_run_config(args)
        self.assertEqual("value", cfg.get("key"))
        self.assertEqual("bar", cfg.get("foo"))


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
