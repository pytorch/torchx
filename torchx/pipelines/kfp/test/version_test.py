#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import unittest
from unittest.mock import patch


class VersionTest(unittest.TestCase):
    def test_can_get_version(self) -> None:
        import torchx.pipelines.kfp

        self.assertIsNotNone(torchx.pipelines.kfp.__version__)

    def test_kfp_1x(self) -> None:
        import torchx.pipelines.kfp

        with patch("kfp.__version__", "2.0.1"):
            with self.assertRaisesRegex(ImportError, "Only kfp version"):
                importlib.reload(torchx.pipelines.kfp)

        with patch("kfp.__version__", "1.5.0"):
            importlib.reload(torchx.pipelines.kfp)
