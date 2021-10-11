# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
from torchx.examples.apps.lightning_classy_vision.model import (
    TinyImageNetModel,
)


class ModelTest(unittest.TestCase):
    def test_basic(self) -> None:
        model = TinyImageNetModel()
        self.assertEqual(len(model.seq), 1)
        out = model(torch.zeros((1, 64, 64)))
        self.assertIsNotNone(out)

    def test_layer_sizes(self) -> None:
        model = TinyImageNetModel(
            layer_sizes=[
                10,
                15,
            ],
        )
        self.assertEqual(len(model.seq), 5)
        out = model(torch.zeros((1, 64, 64)))
        self.assertIsNotNone(out)
