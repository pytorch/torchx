#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Integration tests for KFP v2 adapter that test component creation and pipeline compilation.

This module tests the adapter module that converts TorchX AppDef
to KFP v2 components, focusing on component creation, task configuration,
and pipeline compilation.
"""

import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from kfp import compiler, dsl, local
from torchx import specs
from torchx.pipelines.kfp.adapter import component_from_app, container_from_app


class TestTorchXComponentCreation(unittest.TestCase):
    """Test TorchX component creation and metadata."""

    def test_simple_component_creation(self):
        """Test creating a simple container component from TorchX AppDef."""
        app = specs.AppDef(
            name="echo-test",
            roles=[
                specs.Role(
                    name="worker",
                    entrypoint="/bin/echo",
                    args=["Hello from TorchX"],
                    image="alpine:latest",
                    resource=specs.Resource(cpu=1, memMB=512, gpu=0),
                )
            ],
        )

        component = component_from_app(app)

        # Verify component was created correctly
        self.assertIsNotNone(component)
        self.assertTrue(callable(component))
        self.assertEqual(component._component_human_name, "echo-test-worker")
        self.assertIn("TorchX component", component._component_description)

        # Verify the role is attached
        self.assertTrue(hasattr(component, "_torchx_role"))
        self.assertEqual(component._torchx_role.entrypoint, "/bin/echo")
        self.assertEqual(component._torchx_role.args, ["Hello from TorchX"])
        self.assertEqual(component._torchx_role.image, "alpine:latest")

    def test_component_with_environment_variables(self):
        """Test component creation with environment variables."""
        env_vars = {
            "MODEL_PATH": "/models/bert",
            "BATCH_SIZE": "32",
            "LEARNING_RATE": "0.001",
            "CUDA_VISIBLE_DEVICES": "0,1",
        }

        app = specs.AppDef(
            name="ml-training",
            roles=[
                specs.Role(
                    name="trainer",
                    entrypoint="python",
                    args=["train.py", "--distributed"],
                    env=env_vars,
                    image="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",
                    resource=specs.Resource(cpu=8, memMB=32768, gpu=2),
                )
            ],
        )

        component = component_from_app(app)

        # Verify environment variables are stored
        self.assertEqual(component._torchx_role.env, env_vars)
        self.assertEqual(component._torchx_role.resource.gpu, 2)
        self.assertEqual(component._torchx_role.resource.cpu, 8)

    def test_multi_gpu_component_with_metadata(self):
        """Test component with multiple GPUs and accelerator metadata."""
        app = specs.AppDef(
            name="distributed-training",
            metadata={"accelerator_type": "nvidia-tesla-a100"},
            roles=[
                specs.Role(
                    name="ddp-trainer",
                    entrypoint="torchrun",
                    args=[
                        "--nproc_per_node=4",
                        "--master_port=29500",
                        "train_ddp.py",
                        "--epochs=100",
                    ],
                    image="pytorch/pytorch:latest",
                    resource=specs.Resource(cpu=16, memMB=65536, gpu=4),
                )
            ],
        )

        component = component_from_app(app)

        # Verify multi-GPU configuration
        self.assertEqual(component._torchx_role.resource.gpu, 4)
        self.assertEqual(app.metadata["accelerator_type"], "nvidia-tesla-a100")

    def test_component_with_ui_metadata(self):
        """Test component with UI metadata for visualization."""
        ui_metadata = {
            "outputs": [
                {
                    "type": "tensorboard",
                    "source": "gs://my-bucket/tensorboard-logs",
                },
                {
                    "type": "markdown",
                    "storage": "inline",
                    "source": "# Training Complete\nModel saved to gs://my-bucket/model",
                },
            ]
        }

        app = specs.AppDef(
            name="viz-test",
            roles=[
                specs.Role(
                    name="worker",
                    entrypoint="python",
                    args=["visualize.py"],
                    image="python:3.9",
                    resource=specs.Resource(cpu=2, memMB=4096, gpu=0),
                )
            ],
        )

        component = component_from_app(app, ui_metadata=ui_metadata)

        # Component should be created successfully with UI metadata
        self.assertIsNotNone(component)


class TestPipelineCompilation(unittest.TestCase):
    """Test pipeline compilation with TorchX components."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Clean up after tests."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_compile_simple_pipeline(self):
        """Test compiling a pipeline with TorchX components."""
        app1 = specs.AppDef(
            name="preprocessor",
            roles=[
                specs.Role(
                    name="worker",
                    entrypoint="python",
                    args=[
                        "preprocess.py",
                        "--input=/data/raw",
                        "--output=/data/processed",
                    ],
                    image="python:3.9",
                    resource=specs.Resource(cpu=2, memMB=4096, gpu=0),
                )
            ],
        )

        app2 = specs.AppDef(
            name="trainer",
            roles=[
                specs.Role(
                    name="worker",
                    entrypoint="python",
                    args=["train.py", "--data=/data/processed"],
                    image="pytorch/pytorch:latest",
                    resource=specs.Resource(cpu=4, memMB=8192, gpu=1),
                )
            ],
        )

        @dsl.pipeline(
            name="torchx-pipeline", description="Pipeline with TorchX components"
        )
        def torchx_pipeline():
            # Create tasks from TorchX apps
            preprocess_task = container_from_app(
                app1, display_name="Data Preprocessing", enable_caching=True
            )

            train_task = container_from_app(
                app2,
                display_name="Model Training",
                retry_policy={"max_retry_count": 2},
                accelerator_type="nvidia-tesla-v100",
            )

            # Set task dependencies
            train_task.after(preprocess_task)

        # Compile the pipeline
        output_path = os.path.join(self.temp_dir, "pipeline.yaml")
        compiler.Compiler().compile(
            pipeline_func=torchx_pipeline, package_path=output_path
        )

        # Verify the pipeline was compiled
        self.assertTrue(os.path.exists(output_path))

        # Read and verify pipeline structure
        with open(output_path) as f:
            pipeline_content = f.read()

        # Check that key components are in the pipeline
        self.assertIn("torchx-component", pipeline_content)
        self.assertIn("python:3.9", pipeline_content)
        self.assertIn("pytorch/pytorch:latest", pipeline_content)
        self.assertIn("Data Preprocessing", pipeline_content)
        self.assertIn("Model Training", pipeline_content)

    def test_compile_ml_pipeline_with_parameters(self):
        """Test compiling a complete ML pipeline with parameters."""

        @dsl.pipeline(
            name="ml-training-pipeline",
            description="Complete ML pipeline with parameters",
        )
        def ml_pipeline(
            learning_rate: float = 0.001,
            batch_size: int = 32,
            epochs: int = 50,
            gpu_type: str = "nvidia-tesla-v100",
        ):
            # Preprocessing step
            preprocess_app = specs.AppDef(
                name="preprocess",
                roles=[
                    specs.Role(
                        name="preprocessor",
                        entrypoint="python",
                        args=["preprocess_data.py", "--batch-size", str(batch_size)],
                        image="python:3.9-slim",
                        resource=specs.Resource(cpu=4, memMB=16384, gpu=0),
                    )
                ],
            )

            preprocess_task = container_from_app(
                preprocess_app, display_name="Data Preprocessing", enable_caching=True
            )

            # Training step
            train_app = specs.AppDef(
                name="train",
                roles=[
                    specs.Role(
                        name="trainer",
                        entrypoint="python",
                        args=[
                            "train_model.py",
                            f"--learning-rate={learning_rate}",
                            f"--batch-size={batch_size}",
                            f"--epochs={epochs}",
                        ],
                        env={"CUDA_VISIBLE_DEVICES": "0,1"},
                        image="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",
                        resource=specs.Resource(cpu=8, memMB=32768, gpu=2),
                    )
                ],
            )

            train_task = container_from_app(
                train_app,
                display_name=f"Model Training (LR={learning_rate})",
                retry_policy={
                    "max_retry_count": 3,
                    "backoff_duration": "300s",
                    "backoff_factor": 2.0,
                },
                accelerator_type=gpu_type,
            )
            train_task.after(preprocess_task)

            # Evaluation step
            eval_app = specs.AppDef(
                name="evaluate",
                roles=[
                    specs.Role(
                        name="evaluator",
                        entrypoint="python",
                        args=["evaluate_model.py"],
                        image="pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime",
                        resource=specs.Resource(cpu=4, memMB=16384, gpu=1),
                    )
                ],
            )

            eval_task = container_from_app(
                eval_app, display_name="Model Evaluation", enable_caching=False
            )
            eval_task.after(train_task)

        # Compile the pipeline
        output_path = os.path.join(self.temp_dir, "ml_pipeline.yaml")
        compiler.Compiler().compile(pipeline_func=ml_pipeline, package_path=output_path)

        # Verify pipeline was compiled
        self.assertTrue(os.path.exists(output_path))

        # Read and verify content
        with open(output_path) as f:
            content = f.read()

        # Verify all components and parameters are present
        self.assertIn("Data Preprocessing", content)
        self.assertIn("Model Training", content)
        self.assertIn("Model Evaluation", content)
        self.assertIn("learning_rate", content)
        self.assertIn("batch_size", content)
        self.assertIn("epochs", content)
        self.assertIn("gpu_type", content)

        # Verify resource configurations
        self.assertIn("resourceCpuLimit", content)
        self.assertIn("resourceMemoryLimit", content)
        self.assertIn("accelerator", content)

    def test_compile_nested_pipeline(self):
        """Test compiling a pipeline with nested components."""
        simple_app = specs.AppDef(
            name="worker",
            roles=[
                specs.Role(
                    name="task",
                    entrypoint="echo",
                    args=["Processing"],
                    image="alpine:latest",
                    resource=specs.Resource(cpu=1, memMB=512, gpu=0),
                )
            ],
        )

        @dsl.pipeline(name="inner-pipeline")
        def inner_pipeline(message: str):
            task1 = container_from_app(simple_app, display_name=f"Task 1: {message}")
            task2 = container_from_app(simple_app, display_name=f"Task 2: {message}")
            task2.after(task1)

        @dsl.pipeline(name="outer-pipeline")
        def outer_pipeline():
            # Preprocessing
            preprocessing = container_from_app(simple_app, display_name="Preprocessing")

            # Inner pipeline
            inner = inner_pipeline(message="Inner Processing")
            inner.after(preprocessing)

            # Postprocessing
            postprocessing = container_from_app(
                simple_app, display_name="Postprocessing"
            )
            postprocessing.after(inner)

        # Compile the pipeline
        output_path = os.path.join(self.temp_dir, "nested_pipeline.yaml")
        compiler.Compiler().compile(
            pipeline_func=outer_pipeline, package_path=output_path
        )

        # Verify compilation
        self.assertTrue(os.path.exists(output_path))

        # Verify structure
        with open(output_path) as f:
            content = f.read()

        self.assertIn("Preprocessing", content)
        self.assertIn("Inner Processing", content)
        self.assertIn("Postprocessing", content)


class TestLocalExecution(unittest.TestCase):
    """Test local execution of lightweight Python components.

    Note: Container components require DockerRunner which may not be available
    in all test environments, so we focus on testing with lightweight Python
    components to verify the execution flow.
    """

    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        # Initialize local execution with SubprocessRunner
        local.init(runner=local.SubprocessRunner(), pipeline_root=self.temp_dir)

    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_simple_component_execution(self):
        """Test executing a simple Python component."""

        @dsl.component(base_image="python:3.9-slim")
        def add_numbers(a: float, b: float) -> float:
            """Add two numbers."""
            return a + b

        # Execute component
        task = add_numbers(a=5.0, b=3.0)

        # Verify result
        self.assertEqual(task.output, 8.0)

    def test_component_with_artifact_output(self):
        """Test component that produces output artifacts."""

        @dsl.component(base_image="python:3.9-slim")
        def generate_report(data: dict, report_name: str) -> str:
            """Generate a report from data."""
            import json

            report = {
                "name": report_name,
                "data": data,
                "summary": f"Report contains {len(data)} items",
            }

            return json.dumps(report)

        # Execute component
        test_data = {"metric1": 0.95, "metric2": 0.87}
        task = generate_report(data=test_data, report_name="Test Report")

        # Verify output
        result = json.loads(task.output)
        self.assertEqual(result["name"], "Test Report")
        self.assertEqual(result["data"], test_data)
        self.assertIn("2 items", result["summary"])

    def test_pipeline_execution(self):
        """Test executing a pipeline with multiple components."""

        @dsl.component(base_image="python:3.9-slim")
        def preprocess(value: float) -> float:
            """Preprocess input value."""
            return value * 2.0

        @dsl.component(base_image="python:3.9-slim")
        def transform(value: float, factor: float = 1.5) -> float:
            """Transform value by factor."""
            return value * factor

        @dsl.pipeline(name="test-pipeline")
        def data_pipeline(input_value: float = 10.0) -> float:
            prep_task = preprocess(value=input_value)
            trans_task = transform(value=prep_task.output, factor=3.0)
            return trans_task.output

        # Execute pipeline
        pipeline_task = data_pipeline(input_value=5.0)

        # Verify result: 5.0 * 2.0 * 3.0 = 30.0
        self.assertEqual(pipeline_task.output, 30.0)

    def test_conditional_execution(self):
        """Test conditional execution in a pipeline."""

        @dsl.component(base_image="python:3.9-slim")
        def check_threshold(value: float, threshold: float = 0.5) -> str:
            """Check if value exceeds threshold."""
            return "high" if value > threshold else "low"

        @dsl.component(base_image="python:3.9-slim")
        def process_high(value: float) -> float:
            """Process high values."""
            return value * 2.0

        @dsl.component(base_image="python:3.9-slim")
        def process_low(value: float) -> float:
            """Process low values."""
            return value * 0.5

        # Test with high value
        check_task = check_threshold(value=0.8)
        self.assertEqual(check_task.output, "high")

        # Test with low value
        check_task = check_threshold(value=0.3)
        self.assertEqual(check_task.output, "low")

        # Test processing based on condition
        high_task = process_high(value=10.0)
        self.assertEqual(high_task.output, 20.0)

        low_task = process_low(value=10.0)
        self.assertEqual(low_task.output, 5.0)


if __name__ == "__main__":
    unittest.main()
