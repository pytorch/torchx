#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from pathlib import Path

from torchx.cli.main import main


_root: Path = Path(__file__).parent

_SIMPLE_CONF: str = str(Path(__file__).parent / "components.py:simple")


class CLITest(unittest.TestCase):
    def setUp(self) -> None:
        self.old_cwd = os.getcwd()
        os.chdir(_root / "container")

    def tearDown(self) -> None:
        os.chdir(self.old_cwd)

    def test_run_abs_config_path(self) -> None:
        main(
            [
                "run",
                "--scheduler",
                "local_cwd",
                str(_root / "components.py:simple"),
                "--num_trainers",
                "2",
            ]
        )

    def test_run_builtin_config(self) -> None:
        main(
            [
                "run",
                "--scheduler",
                "local_cwd",
                _SIMPLE_CONF,
                "--num_trainers",
                "2",
            ]
        )
