# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import examples.apps.dist_cifar.component as dist_cifar
from torchx.components.component_test_base import ComponentTestCase


class DistCifar10ComponentTest(ComponentTestCase):
    def test_trainer(self) -> None:
        self._validate(dist_cifar, "trainer")
