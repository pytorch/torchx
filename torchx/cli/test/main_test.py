#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from pathlib import Path

from torchx.cli.main import main

_root: Path = Path(__file__).parent


class CLITest(unittest.TestCase):
    def test_run(self) -> None:
        with self.assertRaises(NotImplementedError):
            main(
                [
                    "run",
                    str(_root / "examples" / "simple.conf"),
                    "--num_trainers",
                    "10",
                ]
            )
