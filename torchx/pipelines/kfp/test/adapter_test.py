#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os.path
import tempfile
import unittest
from typing import List, Callable

import torchx
import yaml
from kfp import compiler, components, dsl
from kubernetes.client.models import V1ContainerPort, V1ResourceRequirements
from torchx.pipelines.kfp.adapter import (
    component_from_app,
    component_spec_from_app,
    container_from_app,
    ContainerFactory,
)
from torchx.specs import api


class KFPSpecsTest(unittest.TestCase):
    """
    tests KFP components using torchx.specs.api
    """

    def _test_app(self) -> api.AppDef:
        trainer_role = api.Role(
            name="trainer",
            image="pytorch/torchx:latest",
            entrypoint="main",
            args=["--output-path", "blah"],
            env={"FOO": "bar"},
            resource=api.Resource(
                cpu=2,
                memMB=3000,
                gpu=4,
            ),
            port_map={"foo": 1234},
            num_replicas=1,
        )

        return api.AppDef("test", roles=[trainer_role])

    def _compile_pipeline(self, pipeline: Callable[[], None]) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline_file = os.path.join(tmpdir, "pipeline.yaml")
            compiler.Compiler().compile(pipeline, pipeline_file)
            with open(pipeline_file, "r") as f:
                data = yaml.safe_load(f)

        spec = data["spec"]
        templates = spec["templates"]
        self.assertGreaterEqual(len(templates), 2)

    def test_component_spec_from_app(self) -> None:
        app = self._test_app()

        spec, role = component_spec_from_app(app)
        self.assertIsNotNone(components.load_component_from_text(spec))
        self.assertEqual(role.resource, app.roles[0].resource)
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
outputs: []
""",
        )

    def test_pipeline(self) -> None:
        app = self._test_app()
        kfp_copy: ContainerFactory = component_from_app(app)

        def pipeline() -> None:
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

        self._compile_pipeline(pipeline)

    def test_pipeline_metadata(self) -> None:
        app = self._test_app()
        metadata = {}
        kfp_copy: ContainerFactory = component_from_app(app, metadata)

        def pipeline() -> None:
            a = kfp_copy()
            self.assertEqual(len(a.volumes), 1)
            self.assertEqual(len(a.container.volume_mounts), 1)
            self.assertEqual(len(a.sidecars), 1)
            self.assertEqual(
                a.output_artifact_paths["mlpipeline-ui-metadata"],
                "/tmp/outputs/mlpipeline-ui-metadata/data.json",
            )
            self.assertEqual(
                a.pod_labels,
                {
                    "torchx.pytorch.org/version": torchx.__version__,
                    "torchx.pytorch.org/app-name": "test",
                    "torchx.pytorch.org/role-index": "0",
                    "torchx.pytorch.org/role-name": "trainer",
                    "torchx.pytorch.org/replica-id": "0",
                },
            )

        self._compile_pipeline(pipeline)

    def test_container_from_app(self) -> None:
        app: api.AppDef = self._test_app()

        def pipeline() -> None:
            a: dsl.ContainerOp = container_from_app(app)
            b: dsl.ContainerOp = container_from_app(app)
            b.after(a)

        self._compile_pipeline(pipeline)
