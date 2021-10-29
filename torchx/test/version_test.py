#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest


class VersionTest(unittest.TestCase):
    def test_can_get_version(self) -> None:
        import torchx

        self.assertIsNotNone(torchx.__version__)
        self.assertIsNotNone(torchx.IMAGE)

    def test_images(self) -> None:
        from torchx.version import __version__, TORCHX_IMAGE, EXAMPLES_IMAGE

        self.assertEqual(TORCHX_IMAGE, f"ghcr.io/pytorch/torchx:{__version__}")
        self.assertEqual(
            EXAMPLES_IMAGE, f"ghcr.io/pytorch/torchx-examples:{__version__}"
        )
