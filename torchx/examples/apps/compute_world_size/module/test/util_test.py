# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import random
import unittest

from omegaconf import DictConfig
from torchx.examples.apps.compute_world_size.module.util import compute_world_size


class UtilTest(unittest.TestCase):
    def test_compute_world_size(self) -> None:
        cfg = DictConfig(
            content={
                "main": {
                    "rank": 0,
                    "world_size": 1,
                    "master_addr": "localhost",
                    # ephemeral port range in linux
                    "master_port": random.randint(32768, 60999),
                    "backend": "gloo",
                }
            }
        )

        self.assertEqual(1, compute_world_size(cfg))
