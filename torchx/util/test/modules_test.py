# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchx.util.modules import load_module


class ModulesTest(unittest.TestCase):
    def test_load_module(self) -> None:
        result = load_module("os.path")
        import os

        self.assertEqual(result, os.path)

    def test_load_module_method(self) -> None:
        result = load_module("os.path:join")
        import os

        self.assertEqual(result, os.path.join)
