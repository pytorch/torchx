#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import importlib
import unittest
from unittest.mock import patch
import warnings


class VersionTest(unittest.TestCase):
    def test_can_get_version(self) -> None:
        import torchx.pipelines.kfp

        self.assertIsNotNone(torchx.pipelines.kfp.__version__)

    def test_kfp_1x(self) -> None:
        import torchx.pipelines.kfp

        # KFP 2.x should not trigger any warnings
        with patch("kfp.__version__", "2.0.1"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                importlib.reload(torchx.pipelines.kfp)
                self.assertEqual(len(w), 0)

        # KFP 1.x should trigger a DeprecationWarning
        with patch("kfp.__version__", "1.5.0"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                importlib.reload(torchx.pipelines.kfp)
                self.assertEqual(len(w), 1)
                self.assertTrue(issubclass(w[-1].category, DeprecationWarning))
                self.assertIn("KFP version 1.x.x is deprecated", str(w[-1].message))
