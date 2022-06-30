#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import unittest
from unittest.mock import MagicMock, patch

from torchx.cli.cmd_list import CmdList


class CmdListTest(unittest.TestCase):
    @patch("torchx.runner.api.Runner.list")
    def test_run(self, list: MagicMock) -> None:
        parser = argparse.ArgumentParser()
        cmd_list = CmdList()
        cmd_list.add_arguments(parser)

        args = parser.parse_args(
            [
                "--scheduler",
                "kubernetes",
            ]
        )
        cmd_list.run(args)

        self.assertEqual(list.call_count, 1)
        list.assert_called_with("kubernetes")
