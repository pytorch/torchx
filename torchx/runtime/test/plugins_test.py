# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
import unittest

import yaml
from torchx.runtime.plugins import init_plugins, TORCHX_CONFIG_ENV


class ContainerTest(unittest.TestCase):
    def test_config_plugins(self) -> None:
        """
        Tests that storage providers from the specified config are loaded.
        """
        from torchx.runtime.test import dummy_module

        module = "torchx.runtime.test.dummy_module"
        config = {
            "plugins": {
                module: {
                    "foo": "bar",
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "torchx.yaml")
            with open(config_path, "w") as f:
                yaml.dump(config, f)
            os.environ[TORCHX_CONFIG_ENV] = config_path
            self.assertEqual(dummy_module.INIT_COUNT, 0)
            init_plugins()
            self.assertEqual(dummy_module.INIT_COUNT, 1)
