# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import tempfile
import unittest
from argparse import ArgumentParser
from pathlib import Path
from unittest import mock

from torchx.cli import argparse_util
from torchx.cli.argparse_util import scheduler_params, torchxconfig_run


CONFIG_DIRS = "torchx.cli.argparse_util.CONFIG_DIRS"


class ArgparseUtilTest(unittest.TestCase):
    def _write(self, filename: str, content: str) -> Path:
        f = Path(self.test_dir) / filename
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "w") as fp:
            fp.write(content)
        return f

    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp(prefix="torchx_argparse_util_test")
        argparse_util._torchxconfig._subcmd_configs.clear()

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_torchxconfig_action(self) -> None:
        with mock.patch(CONFIG_DIRS, [self.test_dir]):
            self._write(
                ".torchxconfig",
                """
[cli:run]
workspace = baz
                """,
            )

            parser = ArgumentParser()

            subparsers = parser.add_subparsers()
            run_parser = subparsers.add_parser("run")

            run_parser.add_argument(
                "--workspace",
                default="foo",
                type=str,
                action=torchxconfig_run,
            )

            # arguments specified in CLI should take outmost precedence
            args = parser.parse_args(["run", "--workspace", "bar"])
            self.assertEqual("bar", args.workspace)

            # if not specified in CLI, then grab it from .torchxconfig
            args = parser.parse_args(["run"])
            self.assertEqual("baz", args.workspace)

    def test_torchxconfig_action_argparse_default(self) -> None:
        with mock.patch(CONFIG_DIRS, [self.test_dir]):
            self._write(
                ".torchxconfig",
                """
[cli:run]
                """,
            )

            parser = ArgumentParser()

            subparsers = parser.add_subparsers()
            run_parser = subparsers.add_parser("run")

            run_parser.add_argument(
                "--workspace",
                default="foo",
                type=str,
                action=torchxconfig_run,
            )

            # if not found in .torchxconfig should use argparse default
            args = parser.parse_args(["run"])
            self.assertEqual("foo", args.workspace)

    def test_torchxconfig_action_required(self) -> None:
        with mock.patch(CONFIG_DIRS, [self.test_dir]):
            self._write(
                ".torchxconfig",
                """
[cli:run]
workspace = bazz
                """,
            )

            parser = ArgumentParser()

            subparsers = parser.add_subparsers()
            run_parser = subparsers.add_parser("run")

            run_parser.add_argument(
                "--workspace",
                required=True,
                type=str,
                action=torchxconfig_run,
            )

            # arguments specified in CLI should take outmost precedence
            args = parser.parse_args(["run", "--workspace", "bar"])
            self.assertEqual("bar", args.workspace)

            # if not specified in CLI, then grab it from .torchxconfig
            args = parser.parse_args(["run"])
            self.assertEqual("bazz", args.workspace)

    def test_torchxconfig_action_aliases(self) -> None:
        # for aliases, the config file needs to declare the original arg
        with mock.patch(CONFIG_DIRS, [self.test_dir]):
            self._write(
                ".torchxconfig",
                """
[cli:run]
workspace = baz
                """,
            )

            parser = ArgumentParser()

            subparsers = parser.add_subparsers()
            run_parser = subparsers.add_parser("run")

            run_parser.add_argument(
                "--workspace",
                "--buck-target",
                type=str,
                required=True,
                action=torchxconfig_run,
            )

            args = parser.parse_args(["run"])
            self.assertEqual("baz", args.workspace)


    def test_scheduler_params(self) -> None:
        with mock.patch(CONFIG_DIRS, [self.test_dir]):
            self._write(
                ".torchxconfig",
                """
[scheduler:test_scheduler]
param1 = val1
param2 = val2
                """,
            )
            params = scheduler_params("test_scheduler")
            self.assertEqual("val1", params["param1"])
            self.assertEqual("val2", params["param2"])