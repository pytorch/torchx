#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

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

parser = argparse.ArgumentParser(description="example kfp pipeline")

# %%
# TorchX components are built around images. Depending on what scheduler
# you're using this can vary but for KFP these images are specified as
# docker containers. We have one container for the example apps and one for
# the standard built in apps. If you modify the torchx example code you'll
# need to rebuild the container before launching it on KFP

from torchx.version import TORCHX_IMAGE

parser.add_argument(
    "--image",
    type=str,
    help="docker image to use for the examples apps",
    default=TORCHX_IMAGE,
)

# %%
# Most TorchX components use
# `fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_ to abstract
# away dealing with remote filesystems. This allows the components to take
# paths like `s3://` to make it easy to use cloud storage providers.
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
# Kubernetes cluster with with the service name `torchserve` in the default
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

import sys

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

import os.path

from torchx import specs
from torchx.components.utils import copy

data_path: str = os.path.join(args.output_path, "tiny-imagenet-200.zip")
copy_app: specs.AppDef = copy(
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


from torchx.examples.apps.datapreproc.component import data_preproc

processed_data_path: str = os.path.join(args.output_path, "processed")
datapreproc_app: specs.AppDef = data_preproc(
    image=args.image, output_path=processed_data_path, input_path=data_path
)

# %%
# Next we'll create the trainer component that takes in the training data from the
# previous datapreproc component. We've defined this in a separate component
# file as you normally would.
#
# Having a separate component file allows you to launch your trainer from the
# TorchX CLI via `torchx run` for fast iteration as well as run it from a
# pipeline in an automated fashion.

# make sure examples is on the path
if "__file__" in globals():
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from torchx.examples.apps.lightning_classy_vision.component import trainer

logs_path: str = os.path.join(args.output_path, "logs")
models_path: str = os.path.join(args.output_path, "models")

trainer_app: specs.AppDef = trainer(
    output_path=models_path,
    load_path=args.load_path or "",
    log_path=logs_path,
    data_path=processed_data_path,
    epochs=1,
    image=args.image,
)

# %%
# To have the tensorboard path show up in KFPs UI we need to some metadata so
# KFP knows where to consume the metrics from.
#
# This will get used when we create the KFP container.

from typing import Dict

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

from torchx.components.serve import torchserve

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

from torchx.examples.apps.lightning_classy_vision.component import interpret

interpret_path: str = os.path.join(args.output_path, "interpret")
interpret_app: specs.AppDef = interpret(
    load_path=os.path.join(models_path, "last.ckpt"),
    data_path=processed_data_path,
    output_path=interpret_path,
    image=args.image,
)

# %%
# Pipeline Definition
# ###################
# The last step is to define the actual pipeline using the torchx components via
# the KFP adapter and export the pipeline package that can be uploaded to a KFP
# cluster.
#
# The KFP adapter currently doesn't track the input and outputs so the
# containers need to have their dependencies specified via `.after()`.
#
# We call `.set_tty()` to make the logs from the components more responsive for
# example purposes.

import kfp
from torchx.pipelines.kfp.adapter import container_from_app


def pipeline() -> None:
    # container_from_app creates a KFP container from the TorchX app
    # definition.
    copy = container_from_app(copy_app)
    copy.container.set_tty()

    datapreproc = container_from_app(datapreproc_app)
    datapreproc.container.set_tty()
    datapreproc.after(copy)

    # For the trainer we want to log that UI metadata so you can access
    # tensorboard from the UI.
    trainer = container_from_app(trainer_app, ui_metadata=ui_metadata)
    trainer.container.set_tty()
    trainer.after(datapreproc)

    serve = container_from_app(serve_app)
    serve.container.set_tty()
    serve.after(trainer)

    # Serve and interpret only require the trained model so we can run them
    # in parallel to each other.
    interpret = container_from_app(interpret_app)
    interpret.container.set_tty()
    interpret.after(trainer)


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
