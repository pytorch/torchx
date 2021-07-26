# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest
from types import ModuleType

from torchx.specs.file_linter import validate


class ComponentTestCase(unittest.TestCase):
    def _validate(self, module: ModuleType, function_name: str) -> None:
        file_path = os.path.abspath(module.__file__)
        with open(file_path, "r") as fp:
            file_content = fp.read()
        linter_errors = validate(file_content, file_path, function_name)
        if len(linter_errors) != 0:
            error_msg = ""
            for linter_error in linter_errors:
                error_msg += f"Lint Error: {linter_error.description}\n"
            raise ValueError(error_msg)
        self.assertEquals(0, len(linter_errors))
