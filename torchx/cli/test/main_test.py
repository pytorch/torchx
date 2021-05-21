#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from pathlib import Path

import torchelastic.tsm.driver as tsm
from torchx.cli.cmd_run import _parse_run_config
from torchx.cli.main import main


_root: Path = Path(__file__).parent

_SIMPLE_CONF: str = "simple_example.torchx"


class CLITest(unittest.TestCase):
    def test_run_abs_config_path(self) -> None:
        main(
            [
                "run",
                "--scheduler",
                "local",
                str(_root / "examples" / "test_simple.torchx"),
                "--num_trainers",
                "2",
                "--trainer_image",
                str(_root / "examples" / "container"),
            ]
        )

    def test_run_builtin_config(self) -> None:
        main(
            [
                "run",
                "--scheduler",
                "local",
                _SIMPLE_CONF,
                "--num_trainers",
                "2",
                "--trainer_image",
                str(_root / "examples" / "container"),
            ]
        )

    def test_run_scheduler_args_empty(self) -> None:
        self.assertEqual(_parse_run_config(""), tsm.RunConfig())

    def test_run_scheduler_args_simple(self) -> None:
        self.assertEqual(
            _parse_run_config("a=1,b=2;3;4"),
            tsm.RunConfig(
                cfgs={
                    "a": "1",
                    "b": ["2", "3", "4"],
                }
            ),
        )
