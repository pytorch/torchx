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
distributed operator using the kubernetes/volcano job scheduler. This only works
in Kubernetes KFP clusters with https://volcano.sh/en/docs/ installed on them.
"""

import kfp
from torchx import specs
from torchx.pipelines.kfp.adapter import resource_from_app


def pipeline() -> None:
    # First we define our AppDef for the component, we set
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

    # To convert the TorchX AppDef into a KFP container we use
    # the resource_from_app adapter. This takes generates a KFP Kubernetes
    # resource operator definition from the TorchX app def and instantiates it.
    echo_container: kfp.dsl.BaseOp = resource_from_app(echo_app, queue="default")


# %%
# To generate the pipeline definition file we need to call into the KFP compiler
# with our pipeline function.

kfp.compiler.Compiler().compile(
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
# See the
# `KFP SDK Examples <https://www.kubeflow.org/docs/components/pipelines/tutorials/sdk-examples/#examples>`_
# for more info on launching KFP pipelines.

# %%
# See the :ref:`examples_pipelines/kfp/advanced_pipeline:Advanced KubeFlow Pipelines Example` for how to chain multiple
# components together and use builtin components.


# sphinx_gallery_thumbnail_path = '_static/img/gallery-kfp.png'
