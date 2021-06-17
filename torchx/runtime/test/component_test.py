#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import TypedDict, Optional

from torchx.runtime.component import Component
from torchx.runtime.storage import upload_blob, download_blob, temppath


class Config(TypedDict):
    a: int
    b: Optional[int]


class Inputs(TypedDict):
    input_path: str


class Outputs(TypedDict):
    output_path: str


class Copy(Component[Config, Inputs, Outputs]):
    Version: str = "0.1"

    def run(self, inputs: Inputs, outputs: Outputs) -> None:
        upload_blob(outputs["output_path"], download_blob(inputs["input_path"]))


class ComponentTest(unittest.TestCase):
    def test_basic_component(self) -> None:
        with temppath() as input_path, temppath() as output_path:
            data = b"banana"
            upload_blob(input_path, data)
            c = Copy(input_path=input_path, output_path=output_path, a=10)
            c.run(c.inputs, c.outputs)
            out = download_blob(output_path)
            self.assertEqual(out, data)

    def test_required_fields(self) -> None:
        with self.assertRaisesRegex(TypeError, "missing required argument a"):
            Copy()

    def test_duplicate_fields(self) -> None:
        class Config(TypedDict):
            a: Optional[int]

        class Inputs(TypedDict):
            a: Optional[int]

        class Outputs(TypedDict):
            a: Optional[int]

        class BadComponent(Component[Config, Inputs, Outputs]):
            Version: str = "0.1"

            def run(self, inputs: Inputs, outputs: Outputs) -> None:
                ...

        with self.assertRaisesRegex(TypeError, "duplicate field name a"):
            BadComponent()
