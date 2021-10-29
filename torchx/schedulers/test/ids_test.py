#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from unittest.mock import patch, MagicMock

from torchx.schedulers.ids import (
    make_unique,
    random_uint64,
    random_id,
)


class IdsTest(unittest.TestCase):
    def test_make_unique(self) -> None:
        name = "test"
        self.assertNotEqual(make_unique(name), make_unique(name))

    def test_make_unique_min_len(self) -> None:
        unique_name = make_unique("test")
        # 16 chars in hex is 64 bits
        self.assertTrue(len(unique_name) >= len("test") + 5)
        self.assertTrue(unique_name.startswith("test-"))

    def test_random_uint64(self) -> None:
        self.assertGreater(random_uint64(), 0)
        self.assertNotEqual(random_uint64(), random_uint64())

    def test_random_id(self) -> None:
        ALPHAS = "abcdefghijklmnopqrstuvwxyz"
        v = random_id()
        self.assertIn(v[0], ALPHAS)
        self.assertGreater(len(v), 5)

    @patch("os.urandom", return_value=bytes(range(8)))
    def test_random_id_seed(self, urandom: MagicMock) -> None:
        self.assertEqual(random_id(), "fzfjxlmln9")

    @patch("os.urandom", return_value=bytes(range(8)))
    def test_make_unique_seed(self, urandom: MagicMock) -> None:
        self.assertEqual(make_unique("test"), "test-fzfjxlmln9")
