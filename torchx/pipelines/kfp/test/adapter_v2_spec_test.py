#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for KFP v2 adapter that test component creation and task configuration.

This module tests the adapter module that converts TorchX AppDef
to KFP v2 components, focusing on proper resource allocation,
environment configuration, and pipeline task settings.
"""

import unittest
from unittest import mock

from torchx import specs
from torchx.pipelines.kfp.adapter import component_from_app, container_from_app


class TestComponentCreation(unittest.TestCase):
    """Test basic component creation from TorchX AppDef."""

    def test_simple_component_creation(self):
        """Test creating a basic component."""
        app = specs.AppDef(
            name="test-app",
            roles=[
                specs.Role(
                    name="worker",
                    entrypoint="/bin/echo",
                    args=["hello", "world"],
                    image="alpine:latest",
                    resource=specs.Resource(cpu=1, memMB=512, gpu=0),
                )
            ],
        )

        component = component_from_app(app)

        # Verify component attributes
        self.assertEqual(component._component_human_name, "test-app-worker")
        self.assertIn("TorchX component", component._component_description)
        self.assertTrue(hasattr(component, "_torchx_role"))
        self.assertEqual(component._torchx_role.entrypoint, "/bin/echo")
        self.assertEqual(component._torchx_role.args, ["hello", "world"])
        self.assertEqual(component._torchx_role.image, "alpine:latest")

    def test_component_with_environment_variables(self):
        """Test component creation with environment variables."""
        env_vars = {
            "MODEL_PATH": "/models/bert",
            "BATCH_SIZE": "32",
            "CUDA_VISIBLE_DEVICES": "0,1",
        }

        app = specs.AppDef(
            name="ml-app",
            roles=[
                specs.Role(
                    name="trainer",
                    entrypoint="python",
                    args=["train.py"],
                    env=env_vars,
                    image="pytorch/pytorch:latest",
                    resource=specs.Resource(cpu=4, memMB=8192, gpu=2),
                )
            ],
        )

        component = component_from_app(app)

        # Verify environment variables are preserved
        self.assertEqual(component._torchx_role.env, env_vars)
        self.assertEqual(component._torchx_role.resource.gpu, 2)

    def test_component_with_ui_metadata(self):
        """Test component creation with UI metadata."""
        ui_metadata = {
            "outputs": [
                {
                    "type": "tensorboard",
                    "source": "gs://my-bucket/logs",
                }
            ]
        }

        app = specs.AppDef(
            name="viz-app",
            roles=[
                specs.Role(
                    name="worker",
                    entrypoint="python",
                    args=["visualize.py"],
                    image="python:3.9",
                    resource=specs.Resource(cpu=1, memMB=2048, gpu=0),
                )
            ],
        )

        # Component should be created successfully with UI metadata
        component = component_from_app(app, ui_metadata=ui_metadata)
        self.assertIsNotNone(component)
        self.assertEqual(component._component_human_name, "viz-app-worker")


class TestContainerTaskConfiguration(unittest.TestCase):
    """Test container task configuration from AppDef."""

    def setUp(self):
        """Set up test fixtures."""
        self.base_app = specs.AppDef(
            name="test-app",
            roles=[
                specs.Role(
                    name="worker",
                    entrypoint="python",
                    args=["script.py"],
                    image="python:3.9",
                    resource=specs.Resource(cpu=2, memMB=4096, gpu=0),
                )
            ],
        )

    def test_basic_container_task(self):
        """Test basic container task creation."""
        with mock.patch(
            "torchx.pipelines.kfp.adapter.component_from_app"
        ) as mock_component_fn:
            mock_task = mock.MagicMock()
            mock_component = mock.MagicMock(return_value=mock_task)
            mock_component._torchx_role = self.base_app.roles[0]
            mock_component_fn.return_value = mock_component

            task = container_from_app(self.base_app)

            # Verify component was called
            self.assertEqual(task, mock_task)
            mock_component.assert_called_once()

            # Verify resource settings
            mock_task.set_cpu_request.assert_called_once_with("2")
            mock_task.set_cpu_limit.assert_called_once_with("2")
            mock_task.set_memory_request.assert_called_once_with("4096M")
            mock_task.set_memory_limit.assert_called_once_with("4096M")

    def test_container_task_with_display_name(self):
        """Test container task with custom display name."""
        with mock.patch(
            "torchx.pipelines.kfp.adapter.component_from_app"
        ) as mock_component_fn:
            mock_task = mock.MagicMock()
            mock_component = mock.MagicMock(return_value=mock_task)
            mock_component._torchx_role = self.base_app.roles[0]
            mock_component_fn.return_value = mock_component

            display_name = "My Custom Task"
            task = container_from_app(self.base_app, display_name=display_name)

            mock_task.set_display_name.assert_called_once_with(display_name)

    def test_container_task_with_caching(self):
        """Test container task with caching configuration."""
        with mock.patch(
            "torchx.pipelines.kfp.adapter.component_from_app"
        ) as mock_component_fn:
            mock_task = mock.MagicMock()
            mock_component = mock.MagicMock(return_value=mock_task)
            mock_component._torchx_role = self.base_app.roles[0]
            mock_component_fn.return_value = mock_component

            # Test enabling caching
            task = container_from_app(self.base_app, enable_caching=True)
            mock_task.set_caching_options.assert_called_once_with(enable_caching=True)

            # Reset mock
            mock_task.reset_mock()

            # Test disabling caching
            task = container_from_app(self.base_app, enable_caching=False)
            mock_task.set_caching_options.assert_called_once_with(enable_caching=False)


class TestResourceConfiguration(unittest.TestCase):
    """Test resource configuration for container tasks."""

    def test_memory_size_conversions(self):
        """Test memory size conversion from MB to KFP format."""
        test_cases = [
            (512, "512M"),  # 512 MB
            (1024, "1024M"),  # 1 GB
            (2048, "2048M"),  # 2 GB
            (4096, "4096M"),  # 4 GB
            (8192, "8192M"),  # 8 GB
            (16384, "16384M"),  # 16 GB
            (1536, "1536M"),  # 1.5 GB (non-standard)
            (0, None),  # Zero memory
        ]

        for memMB, expected_str in test_cases:
            with self.subTest(memMB=memMB):
                app = specs.AppDef(
                    name="memory-test",
                    roles=[
                        specs.Role(
                            name="worker",
                            entrypoint="python",
                            args=["script.py"],
                            image="python:3.9",
                            resource=specs.Resource(cpu=1, memMB=memMB, gpu=0),
                        )
                    ],
                )

                with mock.patch(
                    "torchx.pipelines.kfp.adapter.component_from_app"
                ) as mock_component_fn:
                    mock_task = mock.MagicMock()
                    mock_component = mock.MagicMock(return_value=mock_task)
                    mock_component._torchx_role = app.roles[0]
                    mock_component_fn.return_value = mock_component

                    task = container_from_app(app)

                    if expected_str:
                        mock_task.set_memory_request.assert_called_once_with(
                            expected_str
                        )
                        mock_task.set_memory_limit.assert_called_once_with(expected_str)
                    else:
                        mock_task.set_memory_request.assert_not_called()
                        mock_task.set_memory_limit.assert_not_called()

    def test_gpu_configuration(self):
        """Test GPU resource configuration."""
        gpu_configs = [
            (0, None, None),  # No GPU
            (1, "nvidia-tesla-v100", "nvidia-tesla-v100"),  # Single GPU with type
            (2, None, "nvidia-tesla-k80"),  # Multiple GPUs without type (uses default)
            (4, "nvidia-tesla-a100", "nvidia-tesla-a100"),  # Multi-GPU with type
        ]

        for gpu_count, accelerator_type, expected_type in gpu_configs:
            with self.subTest(gpu_count=gpu_count, accelerator_type=accelerator_type):
                app = specs.AppDef(
                    name="gpu-test",
                    roles=[
                        specs.Role(
                            name="worker",
                            entrypoint="python",
                            args=["train.py"],
                            image="pytorch/pytorch:latest",
                            resource=specs.Resource(cpu=4, memMB=8192, gpu=gpu_count),
                        )
                    ],
                )

                with mock.patch(
                    "torchx.pipelines.kfp.adapter.component_from_app"
                ) as mock_component_fn:
                    mock_task = mock.MagicMock()
                    mock_component = mock.MagicMock(return_value=mock_task)
                    mock_component._torchx_role = app.roles[0]
                    mock_component_fn.return_value = mock_component

                    task = container_from_app(app, accelerator_type=accelerator_type)

                    if gpu_count > 0:
                        mock_task.set_accelerator_limit.assert_called_once_with(
                            str(gpu_count)
                        )
                        if expected_type:
                            mock_task.set_accelerator_type.assert_called_once_with(
                                expected_type
                            )
                    else:
                        mock_task.set_accelerator_limit.assert_not_called()
                        mock_task.set_accelerator_type.assert_not_called()

    def test_fractional_cpu_handling(self):
        """Test handling of fractional CPU values."""
        app = specs.AppDef(
            name="cpu-test",
            roles=[
                specs.Role(
                    name="worker",
                    entrypoint="python",
                    args=["script.py"],
                    image="python:3.9",
                    resource=specs.Resource(cpu=1.5, memMB=1024, gpu=0),
                )
            ],
        )

        with mock.patch(
            "torchx.pipelines.kfp.adapter.component_from_app"
        ) as mock_component_fn:
            mock_task = mock.MagicMock()
            mock_component = mock.MagicMock(return_value=mock_task)
            mock_component._torchx_role = app.roles[0]
            mock_component_fn.return_value = mock_component

            task = container_from_app(app)

            # CPU should be truncated to integer (1.5 -> 1)
            mock_task.set_cpu_request.assert_called_once_with("1")
            mock_task.set_cpu_limit.assert_called_once_with("1")


class TestRetryAndErrorHandling(unittest.TestCase):
    """Test retry policies and error handling configurations."""

    def test_retry_policy_configurations(self):
        """Test various retry policy configurations."""
        app = specs.AppDef(
            name="retry-test",
            roles=[
                specs.Role(
                    name="worker",
                    entrypoint="python",
                    args=["script.py"],
                    image="python:3.9",
                    resource=specs.Resource(cpu=1, memMB=1024, gpu=0),
                )
            ],
        )

        retry_configs = [
            ({"max_retry_count": 5}, {"num_retries": 5}),
            (
                {"max_retry_count": 3, "backoff_duration": "30s"},
                {"num_retries": 3, "backoff_duration": "30s"},
            ),
            (
                {
                    "max_retry_count": 2,
                    "backoff_factor": 2.0,
                    "backoff_max_duration": "300s",
                },
                {
                    "num_retries": 2,
                    "backoff_factor": 2.0,
                    "backoff_max_duration": "300s",
                },
            ),
        ]

        for retry_policy, expected_args in retry_configs:
            with self.subTest(retry_policy=retry_policy):
                with mock.patch(
                    "torchx.pipelines.kfp.adapter.component_from_app"
                ) as mock_component_fn:
                    mock_task = mock.MagicMock()
                    mock_component = mock.MagicMock(return_value=mock_task)
                    mock_component._torchx_role = app.roles[0]
                    mock_component_fn.return_value = mock_component

                    task = container_from_app(app, retry_policy=retry_policy)

                    mock_task.set_retry.assert_called_once_with(**expected_args)

    def test_timeout_configuration(self):
        """Test timeout configuration for tasks."""
        # Skip this test - timeout is not currently implemented in container_from_app
        self.skipTest("Timeout configuration not yet implemented in adapter")


class TestEnvironmentVariables(unittest.TestCase):
    """Test environment variable handling."""

    def test_environment_variable_setting(self):
        """Test that environment variables are properly set on tasks."""
        env_vars = {
            "VAR1": "value1",
            "VAR2": "123",
            "VAR3": "true",
            "PATH_VAR": "/usr/local/bin:/usr/bin",
            "EMPTY_VAR": "",
        }

        app = specs.AppDef(
            name="env-app",
            roles=[
                specs.Role(
                    name="worker",
                    entrypoint="python",
                    args=["app.py"],
                    env=env_vars,
                    image="python:3.9",
                    resource=specs.Resource(cpu=1, memMB=1024, gpu=0),
                )
            ],
        )

        with mock.patch(
            "torchx.pipelines.kfp.adapter.component_from_app"
        ) as mock_component_fn:
            mock_task = mock.MagicMock()
            mock_component = mock.MagicMock(return_value=mock_task)
            mock_component._torchx_role = app.roles[0]
            mock_component_fn.return_value = mock_component

            task = container_from_app(app)

            # Verify all environment variables were set
            expected_calls = [
                mock.call(name=name, value=str(value))
                for name, value in env_vars.items()
            ]
            mock_task.set_env_variable.assert_has_calls(expected_calls, any_order=True)
            self.assertEqual(mock_task.set_env_variable.call_count, len(env_vars))

    def test_special_environment_variables(self):
        """Test handling of special environment variables."""
        special_env_vars = {
            "CUDA_VISIBLE_DEVICES": "0,1,2,3",
            "NCCL_DEBUG": "INFO",
            "PYTHONPATH": "/app:/lib",
            "LD_LIBRARY_PATH": "/usr/local/cuda/lib64",
        }

        app = specs.AppDef(
            name="special-env-app",
            roles=[
                specs.Role(
                    name="worker",
                    entrypoint="python",
                    args=["distributed_train.py"],
                    env=special_env_vars,
                    image="pytorch/pytorch:latest",
                    resource=specs.Resource(cpu=8, memMB=32768, gpu=4),
                )
            ],
        )

        with mock.patch(
            "torchx.pipelines.kfp.adapter.component_from_app"
        ) as mock_component_fn:
            mock_task = mock.MagicMock()
            mock_component = mock.MagicMock(return_value=mock_task)
            mock_component._torchx_role = app.roles[0]
            mock_component_fn.return_value = mock_component

            task = container_from_app(app)

            # Verify special environment variables are set correctly
            for name, value in special_env_vars.items():
                mock_task.set_env_variable.assert_any_call(name=name, value=value)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def test_minimal_resource_spec(self):
        """Test handling of minimal resource specifications."""
        app = specs.AppDef(
            name="minimal-app",
            roles=[
                specs.Role(
                    name="worker",
                    entrypoint="echo",
                    args=["test"],
                    image="alpine:latest",
                    resource=specs.Resource(cpu=0, memMB=0, gpu=0),
                )
            ],
        )

        with mock.patch(
            "torchx.pipelines.kfp.adapter.component_from_app"
        ) as mock_component_fn:
            mock_task = mock.MagicMock()
            mock_component = mock.MagicMock(return_value=mock_task)
            mock_component._torchx_role = app.roles[0]
            mock_component_fn.return_value = mock_component

            task = container_from_app(app)

            # Verify no resource methods were called for zero resources
            mock_task.set_cpu_request.assert_not_called()
            mock_task.set_cpu_limit.assert_not_called()
            mock_task.set_memory_request.assert_not_called()
            mock_task.set_memory_limit.assert_not_called()
            mock_task.set_accelerator_type.assert_not_called()
            mock_task.set_accelerator_limit.assert_not_called()

    def test_very_large_resources(self):
        """Test handling of very large resource requests."""
        app = specs.AppDef(
            name="large-app",
            roles=[
                specs.Role(
                    name="worker",
                    entrypoint="python",
                    args=["bigdata.py"],
                    image="python:3.9",
                    resource=specs.Resource(cpu=128, memMB=524288, gpu=8),  # 512 GB RAM
                )
            ],
        )

        with mock.patch(
            "torchx.pipelines.kfp.adapter.component_from_app"
        ) as mock_component_fn:
            mock_task = mock.MagicMock()
            mock_component = mock.MagicMock(return_value=mock_task)
            mock_component._torchx_role = app.roles[0]
            mock_component_fn.return_value = mock_component

            task = container_from_app(app)

            # Verify large resources are set correctly
            mock_task.set_cpu_request.assert_called_once_with("128")
            mock_task.set_cpu_limit.assert_called_once_with("128")
            mock_task.set_memory_request.assert_called_once_with("524288M")
            mock_task.set_memory_limit.assert_called_once_with("524288M")
            mock_task.set_accelerator_limit.assert_called_once_with("8")

    def test_empty_args_and_entrypoint(self):
        """Test component with no args."""
        app = specs.AppDef(
            name="no-args-app",
            roles=[
                specs.Role(
                    name="worker",
                    entrypoint="/app/start.sh",
                    args=[],  # Empty args
                    image="custom:latest",
                    resource=specs.Resource(cpu=1, memMB=1024, gpu=0),
                )
            ],
        )

        component = component_from_app(app)

        # Verify component is created successfully
        self.assertEqual(component._torchx_role.entrypoint, "/app/start.sh")
        self.assertEqual(component._torchx_role.args, [])


if __name__ == "__main__":
    unittest.main()
