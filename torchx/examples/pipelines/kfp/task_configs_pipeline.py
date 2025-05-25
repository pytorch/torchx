#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task Configuration Pipeline Example for KFP v2

This example demonstrates all available task configuration options in KFP v2:
- Display names
- Resource limits (CPU, memory, GPU/accelerator)
- Environment variables
- Retry policies
- Caching options
"""

import argparse
from kfp import dsl, compiler
from torchx import specs
from torchx.pipelines.kfp.adapter import container_from_app


def main(args: argparse.Namespace) -> None:
    # Create various apps to demonstrate different configurations
    
    # Basic CPU task
    cpu_app = specs.AppDef(
        name="cpu-task",
        roles=[
            specs.Role(
                name="worker",
                entrypoint="python",
                args=["-c", "print('CPU task running'); import time; time.sleep(5)"],
                image="python:3.9-slim",
                resource=specs.Resource(cpu=2, memMB=2048, gpu=0),
            )
        ],
    )
    
    # GPU task
    gpu_app = specs.AppDef(
        name="gpu-task",
        roles=[
            specs.Role(
                name="trainer",
                entrypoint="python",
                args=[
                    "-c",
                    "import torch; print(f'GPU available: {torch.cuda.is_available()}'); "
                    "print(f'GPU count: {torch.cuda.device_count()}')"
                ],
                image="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime",
                resource=specs.Resource(cpu=4, memMB=8192, gpu=1),
            )
        ],
    )
    
    # Task with environment variables
    env_app = specs.AppDef(
        name="env-task",
        roles=[
            specs.Role(
                name="worker",
                entrypoint="python",
                args=[
                    "-c",
                    "import os; "
                    "print(f'MODEL_NAME={os.getenv(\"MODEL_NAME\")}'); "
                    "print(f'BATCH_SIZE={os.getenv(\"BATCH_SIZE\")}'); "
                    "print(f'LEARNING_RATE={os.getenv(\"LEARNING_RATE\")}');"
                ],
                env={
                    "MODEL_NAME": "resnet50",
                    "BATCH_SIZE": "32",
                    "LEARNING_RATE": "0.001",
                },
                image="python:3.9-slim",
                resource=specs.Resource(cpu=1, memMB=1024, gpu=0),
            )
        ],
    )
    
    # Task that might fail (for retry demonstration)
    flaky_app = specs.AppDef(
        name="flaky-task",
        roles=[
            specs.Role(
                name="worker",
                entrypoint="python",
                args=[
                    "-c",
                    "import random; import sys; "
                    "success = random.random() > 0.7; "  # 70% failure rate
                    "print(f'Attempt result: {\"SUCCESS\" if success else \"FAILURE\"}'); "
                    "sys.exit(0 if success else 1);"
                ],
                image="python:3.9-slim",
                resource=specs.Resource(cpu=1, memMB=512, gpu=0),
            )
        ],
    )

    @dsl.pipeline(
        name="task-configuration-demo",
        description="Demonstrates all KFP v2 task configuration options"
    )
    def pipeline():
        # Basic CPU task with display name
        cpu_task = container_from_app(
            cpu_app,
            display_name="CPU Processing Task",
            enable_caching=True,
        )
        
        # GPU task with custom accelerator configuration
        gpu_task = container_from_app(
            gpu_app,
            display_name="GPU Training Task",
            enable_caching=False,  # Don't cache GPU tasks
        )
        # Note: GPU settings are automatically applied from the resource spec
        # But you can override the accelerator type if needed:
        # gpu_task.set_accelerator_type('nvidia-tesla-v100')
        
        # Task with environment variables
        env_task = container_from_app(
            env_app,
            display_name="Environment Variables Demo",
        )
        # Add additional runtime environment variables
        env_task.set_env_variable('RUNTIME_VAR', 'runtime_value')
        env_task.set_env_variable('EXPERIMENT_ID', 'exp-001')
        
        # Flaky task with retry policy
        flaky_task = container_from_app(
            flaky_app,
            display_name="Flaky Task with Retries",
            retry_policy={
                'max_retry_count': 5,
                'backoff_duration': '30s',
                'backoff_factor': 2,
                'backoff_max_duration': '300s',
            },
            enable_caching=False,  # Don't cache flaky tasks
        )
        
        # Set task dependencies
        gpu_task.after(cpu_task)
        env_task.after(cpu_task)
        flaky_task.after(gpu_task, env_task)
        
        # Additional task configurations
        
        # Set resource requests/limits explicitly (override defaults)
        cpu_task.set_cpu_request('1')
        cpu_task.set_memory_request('1Gi')
        
        # Chain multiple configurations
        (gpu_task
            .set_env_variable('CUDA_VISIBLE_DEVICES', '0')
            .set_env_variable('TORCH_CUDA_ARCH_LIST', '7.0;7.5;8.0'))

    # Compile the pipeline
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=args.output_path
    )
    print(f"Pipeline compiled to: {args.output_path}")
    
    # Print some helpful information
    print("\nTask Configuration Features Demonstrated:")
    print("1. Display names for better UI visualization")
    print("2. CPU and memory resource requests/limits")
    print("3. GPU/accelerator configuration")
    print("4. Environment variables (from AppDef and runtime)")
    print("5. Retry policies with exponential backoff")
    print("6. Caching control per task")
    print("7. Task dependencies and execution order")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Task configuration demonstration pipeline"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="task_configs_pipeline.yaml",
        help="Path to save the compiled pipeline",
    )
    
    args = parser.parse_args()
    main(args)
