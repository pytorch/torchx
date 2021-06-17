#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
KubeFlow Pipelines Example
==========================

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

from torchx.version import TORCHX_IMAGE, EXAMPLES_IMAGE

parser.add_argument(
    "--image",
    type=str,
    help="docker image to use for the examples apps",
    default=EXAMPLES_IMAGE,
)
parser.add_argument(
    "--torchx_image",
    type=str,
    help="docker image to use for the builtin torchx apps",
    default=TORCHX_IMAGE,
)

# %%
# Most TorchX components use
# `fsspec <https://filesystem-spec.readthedocs.io/en/latest/>`_ to abstract
# away dealing with remote filesystems. This allows the components to take
# paths like `s3://` to make it easy to use cloud storage providers.
parser.add_argument(
    "--data_path",
    type=str,
    help="path to place the data",
    required=True,
)
parser.add_argument("--load_path", type=str, help="checkpoint path to load from")
parser.add_argument(
    "--output_path",
    type=str,
    help="path to place checkpoints and model outputs",
    required=True,
)
parser.add_argument(
    "--log_path", type=str, help="path to place the tensorboard logs", default="/tmp"
)

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

# %%
# Finally, set the output path for the exported KFP pipeline package. This can either be
# .yaml or .zip.
parser.add_argument(
    "--package_path",
    type=str,
    help="path to place the compiled pipeline package",
    default="pipeline.yaml",
)

import sys

args: argparse.Namespace = parser.parse_args(sys.argv[1:])

# %%
# Creating the Components
# #######################
# The first component we're creating is a data preprocessor. TorchX
# separates the definitions (component) from the implementation (app) so in our
# pipeline we just need to define a simple component so TorchX knows how to
# execute the datapreproc app.
#
# datapreproc outputs the data to a specified fsspec path. These paths are all
# specified ahead of time so we have a fully static pipeline.

from torchx import specs
from torchx.components.base.binary_component import binary_component

datapreproc_app: specs.AppDef = binary_component(
    name="examples-datapreproc",
    entrypoint="datapreproc/datapreproc.py",
    args=[
        "--output_path",
        args.data_path,
    ],
    image=args.image,
)

# %%
# Now that we have the TorchX component we need to adapt it so it can run in KFP
# via our KFP adapter. component_from_app takes in a TorchX component and
# returns a KFP component.
from torchx.pipelines.kfp.adapter import ContainerFactory, component_from_app

datapreproc_comp: ContainerFactory = component_from_app(datapreproc_app)

# %%
# Next we'll create the trainer component that takes in the training data from the
# previous datapreproc component.


trainer_app: specs.AppDef = binary_component(
    name="examples-lightning_classy_vision-trainer",
    entrypoint="lightning_classy_vision/train.py",
    args=[
        "--output_path",
        args.output_path,
        "--load_path",
        args.load_path or "",
        "--log_path",
        args.log_path,
        "--data_path",
        args.data_path,
        "--epochs",
        "1",
    ],
    image=args.image,
)

# %%
# To have the tensorboard path show up in KFPs UI we need to some metadata so
# KFP knows where to consume the metrics from.

import os.path
from typing import Dict

ui_metadata: Dict[str, object] = {
    "outputs": [
        {
            "type": "tensorboard",
            "source": os.path.join(args.log_path, "lightning_logs"),
        }
    ]
}
trainer_comp: ContainerFactory = component_from_app(trainer_app, ui_metadata)

# %%
# For the inference, we're leveraging one of the builtin TorchX components. This
# component takes in a model and uploads it to the TorchServe management API
# endpoints.


from torchx.components.serve import torchserve

serve_app: specs.AppDef = torchserve(
    model_path=os.path.join(args.output_path, "model.mar"),
    management_api=args.management_api,
    image=args.torchx_image,
    params={
        "model_name": args.model_name,
        # set this to allocate a worker
        # "initial_workers": 1,
    },
)
serve_comp: ContainerFactory = component_from_app(serve_app)

# %%
# For model interpretability we're leveraging a custom component stored in it's
# own component file. This component takes in the output from datapreproc and
# train components and produces images with integrated gradient results.

# make sure examples is on the path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

from examples.apps.lightning_classy_vision.component import interpret

interpret_app: specs.AppDef = interpret(
    load_path=os.path.join(args.output_path, "last.ckpt"),
    data_path=args.data_path,
    output_path=args.output_path,
    image=args.image,
)
interpret_comp: ContainerFactory = component_from_app(interpret_app)

# %%
# Pipeline Definition
# ###################
# The last step is to define the actual pipeline using the adapted KFP
# components and export the pipeline package that can be uploaded to a KFP
# cluster.
#
# The KFP adapter currently doesn't track the input and outputs so the
# containers need to have their dependencies specified via `.after()`.
#
# We call `.set_tty()` to make the logs from the components more responsive for
# example purposes.

import kfp


def pipeline() -> None:
    datapreproc = datapreproc_comp()
    datapreproc.container.set_tty()

    trainer = trainer_comp()
    trainer.container.set_tty()
    trainer.after(datapreproc)

    serve = serve_comp()
    serve.container.set_tty()
    serve.after(trainer)

    # Serve and interpret only require the trained model so we can run them
    # in parallel to each other.
    interpret = interpret_comp()
    interpret.container.set_tty()
    interpret.after(trainer)


kfp.compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path=args.package_path,
)

# %%
# Once this has all run you should have a pipeline file (typically
# pipeline.yaml) that you can upload to your KFP cluster via the UI or
# a kfp.Client.
