#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

from torchx.components.base.resource_def import list_resources


class ResourceDefTest(unittest.TestCase):
    def test_list_resources(self) -> None:
        resouces = list_resources()
        self.assertEqual(8, len(resouces))
