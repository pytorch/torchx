#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Distributed KubeFlow Pipelines Example
======================================

This is an example KFP pipeline that uses resource_from_app to launch a
distributed job using the kubernetes/volcano job scheduler. This only works
in Kubernetes KFP clusters with https://volcano.sh/en/docs/ installed on them.
"""

from kfp import compiler, dsl
from torchx import specs
from torchx.pipelines.kfp.adapter import resource_from_app


@dsl.pipeline(
    name="distributed-pipeline",
    description="A distributed pipeline using Volcano job scheduler",
)
def pipeline() -> None:
    # First we define our AppDef for the component
    echo_app = specs.AppDef(
        name="test-dist",
        roles=[
            specs.Role(
                name="dist-echo",
                image="alpine",
                entrypoint="/bin/echo",
                args=["hello dist!"],
                num_replicas=3,
            ),
        ],
    )

    # To convert the TorchX AppDef into a KFP v2 task that creates
    # a Volcano job, we use the resource_from_app adapter.
    # This generates a task that uses kubectl to create the Volcano job.
    echo_task = resource_from_app(echo_app, queue="default")

    # Set display name for better visualization
    echo_task.set_display_name("Distributed Echo Job")


# %%
# To generate the pipeline definition file we need to call into the KFP compiler
# with our pipeline function.

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline,
        package_path="pipeline.yaml",
    )

    with open("pipeline.yaml", "rt") as f:
        print(f.read())

# %%
# Once this has all run you should have a pipeline file (typically
# pipeline.yaml) that you can upload to your KFP cluster via the UI or
# a kfp.Client.
#
# Note: In KFP v2, for more advanced Kubernetes resource manipulation,
# consider using the kfp-kubernetes extension library which provides
# better integration with Kubernetes resources.
#
# See the
# `KFP SDK Examples <https://www.kubeflow.org/docs/components/pipelines/user-guides/core-functions/create-a-pipeline-run/>`_
# for more info on launching KFP pipelines.

# %%
# See the :ref:`examples_pipelines/kfp/advanced_pipeline:Advanced KubeFlow Pipelines Example` for how to chain multiple
# components together and use builtin components.


# sphinx_gallery_thumbnail_path = '_static/img/gallery-kfp.png'
