#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Intro KubeFlow Pipelines Example
================================

This an introductory pipeline using KubeFlow Pipelines built with only TorchX
components.

TorchX is intended to allow making cross platform components. As such, we have
a standard definition that uses adapters to convert it to the specific
pipeline platform. This is an example of using the KFP adapter to run a TorchX
component as part of a KubeFlow Pipeline.

TorchX tries to leverage standard mechanisms wherever possible. For KFP we use
the existing KFP pipeline definition syntax and add a single
`component_from_app` conversion step to convert a TorchX component into one
KFP can understand.

Typically you have a separate component file but for this example we define the
AppDef inline.
"""

import kfp
from torchx import specs
from torchx.pipelines.kfp.adapter import container_from_app


def pipeline() -> None:
    # First we define our AppDef for the component. AppDef is a core part of TorchX
    # and can be used to describe complex distributed multi container apps or
    # just a single node component like here.
    echo_app: specs.AppDef = specs.AppDef(
        name="examples-intro",
        roles=[
            specs.Role(
                name="worker",
                entrypoint="/bin/echo",
                args=["Hello TorchX!"],
                image="alpine",
            )
        ],
    )

    # To convert the TorchX AppDef into a KFP container we use
    # the container_from_app adapter. This takes generates a KFP component
    # definition from the TorchX app def and instantiates it into a container.
    echo_container: kfp.dsl.ContainerOp = container_from_app(echo_app)


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
