# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchx.specs.named_resources_generic import NAMED_RESOURCES


class GenericNamedResourcesTest(unittest.TestCase):
    def test_generic_named_resources(self) -> None:
        for resource_name, resource_factory_fn in NAMED_RESOURCES.items():
            resource = resource_factory_fn()
            if resource_name.startswith("gpu"):
                self.assertGreaterEqual(resource.gpu, 1)
            self.assertGreaterEqual(resource.cpu, 1)
            self.assertGreaterEqual(resource.memMB, 1)
            self.assertFalse(resource.capabilities)
            self.assertFalse(resource.devices)
