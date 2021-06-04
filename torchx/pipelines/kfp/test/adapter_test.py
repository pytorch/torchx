#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os.path
import tempfile
import unittest
from typing import Callable, List, Optional, TypedDict

from kfp import compiler, components, dsl
from kubernetes.client.models import V1ContainerPort, V1ResourceRequirements
from torchx.apps.io.copy import Copy
from torchx.pipelines.kfp.adapter import (
    TorchXComponent,
    component_from_app,
    component_spec,
    component_spec_from_app,
)
from torchx.runtime.component import Component
from torchx.specs import api


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
            """description: 'KFP wrapper for TorchX component torchx.pipelines.kfp.test.adapter_test.TestComponent.
  Version: 0.1'
implementation:
  container:
    command:
    - python3
    - torchx/container/main.py
    - torchx.pipelines.kfp.test.adapter_test.TestComponent
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


class KFPSpecsTest(unittest.TestCase):
    """
    tests KFP components using torchx.specs.api
    """

    def _test_app(self) -> api.Application:
        container = api.Container(
            image="pytorch/torchx:latest",
            resources=api.Resource(
                cpu=2,
                memMB=3000,
                gpu=4,
            ),
            port_map={"foo": 1234},
        )
        trainer_role = (
            api.Role(name="trainer")
            .runs(
                "main",
                "--output-path",
                "blah",
                FOO="bar",
            )
            .on(container)
            .replicas(1)
        )

        return api.Application("test").of(trainer_role)

    def test_component_spec_from_app(self) -> None:
        app = self._test_app()

        spec, container = component_spec_from_app(app)
        self.assertIsNotNone(components.load_component_from_text(spec))
        self.assertEqual(container, app.roles[0].container)
        self.assertEqual(
            spec,
            """description: KFP wrapper for TorchX component test, role trainer
implementation:
  container:
    command:
    - main
    - --output-path
    - blah
    env:
      FOO: bar
    image: pytorch/torchx:latest
name: test-trainer
""",
        )

    def test_pipeline(self) -> None:
        app = self._test_app()
        # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
        kfp_copy: Callable = component_from_app(app)

        def pipeline() -> dsl.PipelineParam:
            a = kfp_copy()
            resources: V1ResourceRequirements = a.container.resources
            self.assertEqual(
                resources,
                V1ResourceRequirements(
                    limits={
                        "cpu": "2000m",
                        "memory": "3000M",
                        "nvidia.com/gpu": "4",
                    },
                    requests={
                        "cpu": "2000m",
                        "memory": "3000M",
                    },
                ),
            )
            ports: List[V1ContainerPort] = a.container.ports
            self.assertEqual(
                ports,
                [V1ContainerPort(name="foo", container_port=1234)],
            )

            b = kfp_copy()
            b.after(a)
            return b

        with tempfile.TemporaryDirectory() as tmpdir:
            compiler.Compiler().compile(pipeline, os.path.join(tmpdir, "pipeline.zip"))
