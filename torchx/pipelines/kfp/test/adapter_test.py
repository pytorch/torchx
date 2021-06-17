#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os.path
import tempfile
import unittest
from typing import Callable, List

from kfp import compiler, components, dsl
from kubernetes.client.models import V1ContainerPort, V1ResourceRequirements
from torchx.pipelines.kfp.adapter import (
    component_from_app,
    component_spec_from_app,
)
from torchx.specs import api


class KFPSpecsTest(unittest.TestCase):
    """
    tests KFP components using torchx.specs.api
    """

    def _test_app(self) -> api.AppDef:
        trainer_role = (
            api.Role(
                name="trainer",
                image="pytorch/torchx:latest",
                resource=api.Resource(
                    cpu=2,
                    memMB=3000,
                    gpu=4,
                ),
                port_map={"foo": 1234},
            )
            .runs(
                "main",
                "--output-path",
                "blah",
                FOO="bar",
            )
            .replicas(1)
        )

        return api.AppDef("test").of(trainer_role)

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
