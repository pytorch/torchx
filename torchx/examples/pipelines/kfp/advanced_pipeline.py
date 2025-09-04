#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Advanced KubeFlow Pipelines Example
===================================

This is an example pipeline using KubeFlow Pipelines built with only TorchX
components.

KFP adapters can be used transform the TorchX components directly into
something that can be used within KFP.
"""

# %%
# Input Arguments
# ###############
# Lets first define some arguments for the pipeline.

import argparse
import os.path
import sys
from typing import Dict

import torchx

from kfp import compiler, dsl
from torchx import specs
from torchx.components.dist import ddp as dist_ddp
from torchx.components.serve import torchserve
from torchx.components.utils import copy as utils_copy, python as utils_python
from torchx.pipelines.kfp.adapter import container_from_app

parser = argparse.ArgumentParser(description="example kfp pipeline")

# %%
# TorchX components are built around images. Depending on what scheduler
# you're using this can vary but for KFP these images are specified as
# docker containers. We have one container for the example apps and one for
# the standard built in apps. If you modify the torchx example code you'll
# need to rebuild the container before launching it on KFP


parser.add_argument(
    "--image",
    type=str,
    help="docker image to use for the examples apps",
    default=torchx.IMAGE,
)

# %%
# Most TorchX components use
# `fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_ to abstract
# away dealing with remote filesystems. This allows the components to take
# paths like ``s3://`` to make it easy to use cloud storage providers.
parser.add_argument(
    "--output_path",
    type=str,
    help="path to place the data",
    required=True,
)
parser.add_argument("--load_path", type=str, help="checkpoint path to load from")

# %%
# This example uses the torchserve for inference so we need to specify some
# options. This assumes you have a TorchServe instance running in the same
# Kubernetes cluster with with the service name ``torchserve`` in the default
# namespace.
#
# See https://github.com/pytorch/serve/blob/master/kubernetes/README.md for info
# on how to setup TorchServe.
parser.add_argument(
    "--management_api",
    type=str,
    help="path to the torchserve management API",
    default="http://torchserve.default.svc.cluster.local:8081",
)
parser.add_argument(
    "--model_name",
    type=str,
    help="the name of the inference model",
    default="tiny_image_net",
)

# %% Parse the arguments, you'll need to set these accordingly if running from a
# notebook.


if "NOTEBOOK" in globals():
    argv = [
        "--output_path",
        "/tmp/output",
    ]
else:
    argv = sys.argv[1:]

args: argparse.Namespace = parser.parse_args(argv)

# %%
# Creating the Components
# #######################
# The first step is downloading the data to somewhere we can work on it. For
# this we can just the builtin copy component. This component takes two valid
# fsspec paths and copies them from one to another. In this case we're using
# http as the source and a file under the output_path as the output.


data_path: str = os.path.join(args.output_path, "tiny-imagenet-200.zip")
copy_app: specs.AppDef = utils_copy(
    "http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    data_path,
    image=args.image,
)

# %%
# The next component is for data preprocessing. This takes in the raw data from
# the previous operator and runs some transforms on it for use with the trainer.
#
# datapreproc outputs the data to a specified fsspec path. These paths are all
# specified ahead of time so we have a fully static pipeline.


processed_data_path: str = os.path.join(args.output_path, "processed")
datapreproc_app: specs.AppDef = utils_python(
    "--output_path",
    processed_data_path,
    "--input_path",
    data_path,
    "--limit",
    "100",
    image=args.image,
    m="torchx.examples.apps.datapreproc.datapreproc",
    cpu=1,
    memMB=1024,
)

# %%
# Next we'll create the trainer component that takes in the training data from the
# previous datapreproc component. We've defined this in a separate component
# file as you normally would.
#
# Having a separate component file allows you to launch your trainer from the
# TorchX CLI via ``torchx run`` for fast iteration as well as run it from a
# pipeline in an automated fashion.

# make sure examples is on the path
if "__file__" in globals():
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


logs_path: str = os.path.join(args.output_path, "logs")
models_path: str = os.path.join(args.output_path, "models")

trainer_app: specs.AppDef = dist_ddp(
    *(
        "--output_path",
        models_path,
        "--load_path",
        args.load_path or "",
        "--log_path",
        logs_path,
        "--data_path",
        processed_data_path,
        "--epochs",
        str(1),
    ),
    image=args.image,
    m="torchx.examples.apps.lightning.train",
    j="1x1",
    # per node resource settings
    cpu=1,
    memMB=3000,
)

# %%
# To have the tensorboard path show up in KFPs UI we need to some metadata so
# KFP knows where to consume the metrics from.
#
# This will get used when we create the KFP container.


ui_metadata: Dict[str, object] = {
    "outputs": [
        {
            "type": "tensorboard",
            "source": os.path.join(logs_path, "lightning_logs"),
        }
    ]
}

# %%
# For the inference, we're leveraging one of the builtin TorchX components. This
# component takes in a model and uploads it to the TorchServe management API
# endpoints.


serve_app: specs.AppDef = torchserve(
    model_path=os.path.join(models_path, "model.mar"),
    management_api=args.management_api,
    image=args.image,
    params={
        "model_name": args.model_name,
        # set this to allocate a worker
        # "initial_workers": 1,
    },
)

# %%
# For model interpretability we're leveraging a custom component stored in it's
# own component file. This component takes in the output from datapreproc and
# train components and produces images with integrated gradient results.

interpret_path: str = os.path.join(args.output_path, "interpret")
interpret_app: specs.AppDef = utils_python(
    *(
        "--load_path",
        os.path.join(models_path, "last.ckpt"),
        "--data_path",
        processed_data_path,
        "--output_path",
        interpret_path,
    ),
    image=args.image,
    m="torchx.examples.apps.lightning.interpret",
)

# %%
# Pipeline Definition
# ###################
# The last step is to define the actual pipeline using the torchx components via
# the KFP adapter and export the pipeline package that can be uploaded to a KFP
# cluster.
#
# The KFP adapter currently doesn't track the input and outputs so the
# containers need to have their dependencies specified.
#
# We no longer need to call `.set_tty()` as that was a v1 feature.


@dsl.pipeline(
    name="TorchX Advanced Pipeline",
    description="Advanced KFP pipeline with TorchX components",
)
def pipeline() -> None:
    # container_from_app creates a KFP v2 task from the TorchX app
    # definition.
    copy_task = container_from_app(copy_app)
    copy_task.set_display_name("Download Data")

    datapreproc_task = container_from_app(datapreproc_app)
    datapreproc_task.set_display_name("Preprocess Data")
    # In KFP v2, dependencies are automatically handled based on data flow
    # If you need explicit dependencies, you need to pass outputs as inputs
    datapreproc_task.after(copy_task)

    # For the trainer we want to log that UI metadata so you can access
    # tensorboard from the UI.
    trainer_task = container_from_app(trainer_app, ui_metadata=ui_metadata)
    trainer_task.set_display_name("Train Model")
    trainer_task.after(datapreproc_task)

    if False:
        serve_task = container_from_app(serve_app)
        serve_task.set_display_name("Serve Model")
        serve_task.after(trainer_task)

    if False:
        # Serve and interpret only require the trained model so we can run them
        # in parallel to each other.
        interpret_task = container_from_app(interpret_app)
        interpret_task.set_display_name("Interpret Model")
        interpret_task.after(trainer_task)


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

# sphinx_gallery_thumbnail_path = '_static/img/gallery-kfp.png'
