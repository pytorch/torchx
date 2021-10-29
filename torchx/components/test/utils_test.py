# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchx.components.utils as utils
from torchx.components.component_test_base import ComponentTestCase


class UtilsComponentTest(ComponentTestCase):
    def test_sh(self) -> None:
        self.validate(utils, "sh")

    def test_python(self) -> None:
        self.validate(utils, "python")

    def test_touch(self) -> None:
        self.validate(utils, "touch")

    def test_echo(self) -> None:
        self.validate(utils, "echo")

    def test_copy(self) -> None:
        self.validate(utils, "copy")

    def test_booth(self) -> None:
        self.validate(utils, "booth")
