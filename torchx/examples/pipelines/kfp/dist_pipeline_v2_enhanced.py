#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Enhanced Distributed Pipeline Example for KFP v2

This example demonstrates advanced KFP v2 features including:
- Using kfp-kubernetes for better Kubernetes integration
- Task configuration options (display names, retries, caching)
- Volume mounting for distributed training
- Resource specifications with GPU support
"""

import argparse
from kfp import dsl, compiler
from kfp import kubernetes  # Using kfp-kubernetes extension
from torchx.pipelines.kfp.adapter import container_from_app, resource_from_app
from torchx import specs


def main(args: argparse.Namespace) -> None:
    # Create distributed training app
    ddp_app = specs.AppDef(
        name="distributed-trainer",
        roles=[
            specs.Role(
                name="trainer",
                entrypoint="bash",
                args=[
                    "-c",
                    "echo 'Starting distributed training...'; "
                    "echo 'Node rank: $RANK'; "
                    "echo 'World size: $WORLD_SIZE'; "
                    "python -m torch.distributed.run --nproc_per_node=2 train.py"
                ],
                env={
                    "MASTER_ADDR": "distributed-trainer-0",
                    "MASTER_PORT": "29500",
                },
                num_replicas=3,
                image="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime",
                resource=specs.Resource(
                    cpu=4,
                    memMB=8192,
                    gpu=2,
                ),
            )
        ],
    )
    
    # Create data preprocessing app
    preprocess_app = specs.AppDef(
        name="data-preprocessor",
        roles=[
            specs.Role(
                name="preprocessor",
                entrypoint="python",
                args=["-m", "preprocess", "--input", "/data/raw", "--output", "/data/processed"],
                env={"DATA_FORMAT": "tfrecord"},
                num_replicas=1,
                image="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime",
                resource=specs.Resource(
                    cpu=2,
                    memMB=4096,
                    gpu=0,
                ),
            )
        ],
    )
    
    # Create model evaluation app
    eval_app = specs.AppDef(
        name="model-evaluator",
        roles=[
            specs.Role(
                name="evaluator",
                entrypoint="python",
                args=["-m", "evaluate", "--model", "/models/latest", "--data", "/data/test"],
                env={"METRICS_OUTPUT": "/metrics/eval.json"},
                num_replicas=1,
                image="pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime",
                resource=specs.Resource(
                    cpu=2,
                    memMB=4096,
                    gpu=1,
                ),
            )
        ],
    )

    @dsl.pipeline(
        name="enhanced-distributed-pipeline",
        description="Enhanced distributed ML pipeline with KFP v2 features"
    )
    def pipeline():
        # Create persistent volume for data sharing
        pvc = kubernetes.CreatePVC(
            pvc_name_suffix='-shared-data',
            access_modes=['ReadWriteMany'],
            size='50Gi',
            storage_class_name='standard',
        )
        
        # Data preprocessing step
        preprocess_task = container_from_app(
            preprocess_app,
            display_name="Data Preprocessing",
            retry_policy={
                'max_retry_count': 3,
                'backoff_duration': '60s',
                'backoff_factor': 2,
            },
            enable_caching=True,
        )
        
        # Mount volume for preprocessing
        kubernetes.mount_pvc(
            preprocess_task,
            pvc_name=pvc.outputs['name'],
            mount_path='/data',
        )
        
        # Distributed training using Volcano
        train_task = resource_from_app(
            ddp_app,
            queue="training-queue",
            service_account="ml-training-sa",
            priority_class="high-priority",
        )
        train_task.set_display_name("Distributed Training")
        train_task.after(preprocess_task)
        
        # Model evaluation
        eval_task = container_from_app(
            eval_app,
            display_name="Model Evaluation",
            enable_caching=False,  # Don't cache evaluation results
        )
        eval_task.after(train_task)
        
        # Mount volume for evaluation
        kubernetes.mount_pvc(
            eval_task,
            pvc_name=pvc.outputs['name'],
            mount_path='/data',
        )
        
        # Clean up PVC after evaluation
        delete_pvc = kubernetes.DeletePVC(
            pvc_name=pvc.outputs['name']
        ).after(eval_task)
        delete_pvc.set_display_name("Cleanup Shared Storage")

    # Compile the pipeline
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path=args.output_path
    )
    print(f"Pipeline compiled to: {args.output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced distributed pipeline example")
    parser.add_argument(
        "--output_path",
        type=str,
        default="enhanced_distributed_pipeline.yaml",
        help="Path to save the compiled pipeline",
    )
    
    args = parser.parse_args()
    main(args)
