#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os.path
import tempfile
import unittest
from typing import Callable

import torchx
import yaml
from kfp import compiler, dsl
from torchx.pipelines.kfp.adapter import (
    component_from_app,
    component_spec_from_app,
    container_from_app,
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

    def _compile_pipeline(self, pipeline: Callable[[], None]) -> dict:
        """Compile pipeline and return the compiled structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pipeline_file = os.path.join(tmpdir, "pipeline.yaml")
            compiler.Compiler().compile(
                pipeline_func=pipeline, package_path=pipeline_file
            )
            with open(pipeline_file, "r") as f:
                data = yaml.safe_load(f)
        return data

    def test_component_spec_from_app(self) -> None:
        app = self._test_app()

        # component_spec_from_app is deprecated and returns app name and role
        app_name, role = component_spec_from_app(app)

        # The function should return the app name and first role
        self.assertEqual(app_name, "test")
        self.assertEqual(role, app.roles[0])
        self.assertEqual(role.resource, app.roles[0].resource)
        self.assertEqual(role.name, "trainer")

    def test_pipeline(self) -> None:
        app = self._test_app()

        @dsl.pipeline(name="test-pipeline")
        def pipeline() -> None:
            # Create two instances of the component
            a = container_from_app(app, display_name="Task A")
            b = container_from_app(app, display_name="Task B")
            # Set dependency
            b.after(a)

        # Compile and check structure
        data = self._compile_pipeline(pipeline)

        # KFP v2 compiled pipelines have this structure at root level
        self.assertIn("components", data)
        self.assertIn("deploymentSpec", data)
        self.assertIn("root", data)

        # Check that we have components
        components = data["components"]
        self.assertGreater(len(components), 0)

        # Check executors
        executors = data["deploymentSpec"]["executors"]
        self.assertGreater(len(executors), 0)

        # Check the task structure
        self.assertIn("dag", data["root"])
        self.assertIn("tasks", data["root"]["dag"])

        # We should have 2 tasks
        tasks = data["root"]["dag"]["tasks"]
        self.assertEqual(len(tasks), 2)

        # Check dependency - second task should depend on first
        task_names = list(tasks.keys())
        second_task = tasks[task_names[1]]
        self.assertIn("dependentTasks", second_task)

    def test_pipeline_metadata(self) -> None:
        app = self._test_app()
        ui_metadata = {
            "outputs": [
                {
                    "type": "tensorboard",
                    "source": "gs://my-bucket/logs",
                }
            ]
        }

        @dsl.pipeline(name="test-pipeline-metadata")
        def pipeline() -> None:
            # Create component with UI metadata
            a = container_from_app(
                app, ui_metadata=ui_metadata, display_name="Task with Metadata"
            )

        # Compile pipeline
        data = self._compile_pipeline(pipeline)

        # Check basic structure
        self.assertIn("components", data)
        self.assertIn("deploymentSpec", data)
        self.assertIn("root", data)

        # Check that UI metadata affects the command
        executors = data["deploymentSpec"]["executors"]
        # UI metadata should be present in at least one executor
        found_metadata = False
        for executor in executors.values():
            if "container" in executor:
                command = executor["container"].get("command", [])
                # Check if metadata handling is in the command
                if any("metadata" in str(cmd) for cmd in command):
                    found_metadata = True
                    break
        self.assertTrue(found_metadata, "UI metadata not found in executor commands")

    def test_container_from_app(self) -> None:
        app: api.AppDef = self._test_app()

        @dsl.pipeline(name="test-container-pipeline")
        def pipeline() -> None:
            # Create two tasks from the same app
            a = container_from_app(app, display_name="First Task")
            b = container_from_app(app, display_name="Second Task")
            b.after(a)

        # Compile and verify
        data = self._compile_pipeline(pipeline)
        self.assertIn("root", data)

        # Check tasks
        tasks = data["root"]["dag"]["tasks"]
        self.assertEqual(len(tasks), 2)

        # Check dependency
        # The second task should have a dependency on the first
        task_names = list(tasks.keys())
        second_task = tasks[task_names[1]]
        self.assertIn("dependentTasks", second_task)

        # Check display names
        for task_name, task in tasks.items():
            self.assertIn("taskInfo", task)
            self.assertIn("name", task["taskInfo"])

    def test_resource_configuration(self) -> None:
        """Test that resources are properly configured in the component."""
        app = self._test_app()

        # Create a component and check that it has the right resources
        component = component_from_app(app)

        # The component function should exist
        self.assertIsNotNone(component)

        # Check that the component has the expected metadata
        # In KFP v2, components store metadata differently
        if hasattr(component, "_torchx_role"):
            role = component._torchx_role
            self.assertEqual(role.resource.cpu, 2)
            self.assertEqual(role.resource.memMB, 3000)
            self.assertEqual(role.resource.gpu, 4)

    def test_environment_variables(self) -> None:
        """Test that environment variables are properly passed."""
        app = self._test_app()

        @dsl.pipeline(name="test-env-pipeline")
        def pipeline() -> None:
            task = container_from_app(app)

        # Compile pipeline
        data = self._compile_pipeline(pipeline)

        # Check that the pipeline was compiled successfully
        self.assertIn("deploymentSpec", data)

        # Find the executor and check environment variables
        executors = data["deploymentSpec"]["executors"]
        found_env = False
        for executor_name, executor in executors.items():
            if "container" in executor:
                container = executor["container"]
                if "env" in container:
                    # Check that FOO environment variable is set
                    env_vars = container["env"]
                    for env_var in env_vars:
                        if (
                            env_var.get("name") == "FOO"
                            and env_var.get("value") == "bar"
                        ):
                            found_env = True
                            break
            if found_env:
                break

        self.assertTrue(
            found_env, "Environment variable FOO=bar not found in any executor"
        )


if __name__ == "__main__":
    unittest.main()
