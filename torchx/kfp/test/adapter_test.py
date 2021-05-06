#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os.path
import tempfile
import unittest
from typing import TypedDict, Optional

from kfp import compiler, components, dsl
from torchx.components.io.copy import Copy
from torchx.kfp.adapter import component_spec, TorchXComponent
from torchx.sdk.component import Component


class Config(TypedDict):
    a: int
    b: Optional[int]


class Inputs(TypedDict):
    input_path: str


class Outputs(TypedDict):
    output_path: str


class TestComponent(Component[Config, Inputs, Outputs]):
    Version: str = "0.1"

    def run(self, inputs: Inputs, outputs: Outputs) -> None:
        ...


class KFPTest(unittest.TestCase):
    def test_component_spec(self) -> None:
        self.maxDiff = None
        spec = component_spec(TestComponent)
        self.assertIsNotNone(components.load_component_from_text(spec))
        self.assertEqual(
            spec,
            """description: 'KFP wrapper for TorchX component torchx.kfp.test.adapter_test.TestComponent.
  Version: 0.1'
implementation:
  container:
    command:
    - python3
    - torchx/container/main.py
    - torchx.kfp.test.adapter_test.TestComponent
    - --a
    - inputValue: a
    - --b
    - inputValue: b
    - --input_path
    - inputValue: input_path
    - --output_path
    - inputValue: output_path
    - --output-path-output_path
    - outputPath: output_path
    image: pytorch/torchx:latest
inputs:
- name: a
  type: String
- default: 'null'
  name: b
  type: String
- name: input_path
  type: String
- name: output_path
  type: String
name: TestComponent
outputs:
- name: output_path
  type: String
""",
        )

    def test_pipeline(self) -> None:
        class KFPCopy(TorchXComponent, component=Copy):
            pass

        def pipeline() -> dsl.PipelineParam:
            a = KFPCopy(
                input_path="file:///etc/os-release", output_path="file:///tmp/foo"
            )
            b = KFPCopy(
                input_path=a.outputs["output_path"], output_path="file:///tmp/bar"
            )
            return b.output

        with tempfile.TemporaryDirectory() as tmpdir:
            compiler.Compiler().compile(pipeline, os.path.join(tmpdir, "pipeline.zip"))

    def test_image(self) -> None:
        class KFPCopy(TorchXComponent, component=Copy, image="foo"):
            pass

        copy = KFPCopy(input_path="", output_path="")
        print(copy)
        # pyre-fixme[16]: `KFPCopy` has no attribute `component_ref`.
        self.assertEqual(copy.component_ref.spec.implementation.container.image, "foo")
