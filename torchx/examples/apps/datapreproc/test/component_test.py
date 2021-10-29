# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchx.examples.apps.datapreproc.component as datapreproc
from torchx.components.component_test_base import ComponentTestCase


class DatapreprocComponentTest(ComponentTestCase):
    def test_trainer(self) -> None:
        self.validate(datapreproc, "data_preproc")
