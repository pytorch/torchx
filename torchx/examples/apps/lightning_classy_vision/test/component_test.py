# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchx.examples.apps.lightning_classy_vision.component as lightning_classy_vision
from torchx.components.component_test_base import ComponentTestCase


class DistributedComponentTest(ComponentTestCase):
    def test_trainer(self) -> None:
        self.validate(lightning_classy_vision, "trainer")

    def test_interpret(self) -> None:
        self.validate(lightning_classy_vision, "interpret")
