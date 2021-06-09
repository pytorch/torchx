# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchx.components.base.binary_component import binary_component
from torchx.specs import api


class BinaryComponentTest(unittest.TestCase):
    def test_binary_component(self) -> None:
        want = api.AppDef(
            name="datapreproc",
            roles=[
                api.Role(
                    name="datapreproc",
                    image="pytorch/pytorch:latest",
                    entrypoint="python3",
                    args=["--version"],
                    env={"FOO": "bar"},
                ),
            ],
        )
        out = binary_component(
            name="datapreproc",
            image="pytorch/pytorch:latest",
            entrypoint="python3",
            args=["--version"],
            env={
                "FOO": "bar",
            },
        )
        self.assertEqual(out, want)
