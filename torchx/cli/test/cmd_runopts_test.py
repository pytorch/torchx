#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import unittest

from torchx.cli.cmd_runopts import CmdRunopts
from torchx.schedulers import get_schedulers


class CmdRunOptsTest(unittest.TestCase):
    def test_run(self) -> None:
        # nothing to assert, just make sure the supported schedulers print runopts

        parser = argparse.ArgumentParser()
        cmd_runopts = CmdRunopts()
        cmd_runopts.add_arguments(parser)

        schedulers = get_schedulers(session_name="test").keys()
        test_configs = [[]] + [[scheduler] for scheduler in schedulers]
        for scheduler in test_configs:
            with self.subTest(scheduler=scheduler):
                args = parser.parse_args(scheduler)
                cmd_runopts.run(args)
