#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import unittest
from unittest.mock import MagicMock, patch

from torchx.cli.cmd_cancel import CmdCancel


class CmdCancelTest(unittest.TestCase):
    @patch("torchx.runner.api.Runner.stop")
    def test_run(self, stop: MagicMock) -> None:
        parser = argparse.ArgumentParser()
        cmd_runopts = CmdCancel()
        cmd_runopts.add_arguments(parser)

        args = parser.parse_args(["foo://session/id"])
        cmd_runopts.run(args)

        self.assertEqual(stop.call_count, 1)
        stop.assert_called_with("foo://session/id")
