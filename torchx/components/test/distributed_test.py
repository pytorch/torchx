# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from types import ModuleType

import torchx.components.distributed as distributed_components
from torchx.specs.file_linter import validate


class DistributedComponentTest(unittest.TestCase):
    def _validate(self, module: ModuleType, function_name: str) -> None:
        file_path = path = os.path.abspath(distributed_components.__file__)
        with open(file_path, "r") as fp:
            file_content = fp.read()
        linter_errors = validate(file_content, file_path, function_name)
        self.assertEquals(0, len(linter_errors))

    def test_ddp(self) -> None:
        self._validate(distributed_components, "ddp")
