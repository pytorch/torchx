# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Trainer Component Example
=========================

This is a component definition that runs the example lightning_classy_vision app.
"""

from typing import Optional, Dict

import torchx.specs.api as torchx
from torchx.specs import macros, named_resources, Resource


def trainer(
        image: str,
        output_path: str,
        data_path: Optional[str] = None,
        load_path: str = "",
        log_path: str = "/logs",
        resource: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        skip_export: bool = False,
        epochs: int = 1,
) -> torchx.AppDef:
    """Runs the example lightning_classy_vision app.

    Args:
        image: image to run (e.g. foobar:latest)
        output_path: output path for model checkpoints (e.g. file:///foo/bar)
        load_path: path to load pretrained model from
        data_path: path to the data to load, if data_path is not provided,
            auto generated test data will be used
        log_path: path to save tensorboard logs to
        resource: the resources to use
        env: env variables for the app
        skip_export: disable model export
        epochs: number of epochs to run
    """
    env = env or {}
    args = [
        "-m",
        "torchx.examples.apps.lightning_classy_vision.train",
        "--output_path",
        output_path,
        "--load_path",
        load_path,
        "--log_path",
        log_path,
        "--epochs",
        str(epochs),
    ]
    if data_path:
        args += ["--data_path", data_path]
    if skip_export:
        args.append("--skip_export")
    return torchx.AppDef(
        name="cv-trainer",
        roles=[
            torchx.Role(
                name="worker",
                entrypoint="python",
                args=args,
                env=env,
                image=image,
                resource=named_resources[resource]
                if resource
                else torchx.Resource(cpu=1, gpu=0, memMB=3000),
            )
        ],
    )


def trainer_dist(
        image: str,
        output_path: str,
        data_path: Optional[str] = None,
        load_path: str = "",
        log_path: str = "/logs",
        resource: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        skip_export: bool = False,
        epochs: int = 1,
        nnodes: int = 1,
        nproc_per_node: int = 1,
        rdzv_backend: str = "etcd",
        rdzv_endpoint: str = "etcd-server:2379",
) -> torchx.AppDef:
    """Runs the example lightning_classy_vision app.

    Args:
        image: image to run (e.g. foobar:latest)
        output_path: output path for model checkpoints (e.g. file:///foo/bar)
        load_path: path to load pretrained model from
        data_path: path to the data to load, if data_path is not provided,
            auto generated test data will be used
        log_path: path to save tensorboard logs to
        resource: the resources to use
        env: env variables for the app
        skip_export: disable model export
        epochs: number of epochs to run
        nnodes: number of nodes to run train on, default 1
        nproc_per_node: number of processes per node. Each process
            is assumed to use a separate GPU, default 1
        rdzv_backend: rendezvous backend to use, allowed values can be found at
            https://github.com/pytorch/pytorch/blob/master/torch/distributed/elastic/rendezvous/registry.py
            The default backend is `c10d`
        rdzv_endpoint: Controller endpoint. In case of rdzv_backend is etcd, this is a etcd
            endpoint, in case of c10d, this is the endpoint of one of the hosts.
            The default endpoint is `localhost:30001`
    """
    env = env or {}
    args = [
        "-m",
        "torch.distributed.run",
        "--rdzv_backend",
        rdzv_backend,
        "--rdzv_endpoint",
        rdzv_endpoint,
        "--rdzv_id",
        f"{macros.app_id}",
        "--nnodes",
        str(nnodes),
        "--nproc_per_node",
        str(nproc_per_node),
        "-m",
        "torchx.examples.apps.lightning_classy_vision.train",
        "--output_path",
        output_path,
        "--load_path",
        load_path,
        "--log_path",
        log_path,
        "--epochs",
        str(epochs),
    ]
    if data_path:
        args += ["--data_path", data_path]
    if skip_export:
        args.append("--skip_export")
    resource_def = (
        named_resources[resource]
        if resource
        else Resource(cpu=nnodes, gpu=0, memMB=3000)
    )
    return torchx.AppDef(
        name="dist-cv-trainer",
        roles=[
            torchx.Role(
                name="worker",
                entrypoint="python",
                args=args,
                env=env,
                image=image,
                resource=resource_def,
                num_replicas=nnodes,
            )
        ],
    )


def interpret(
        image: str,
        load_path: str,
        data_path: str,
        output_path: str,
        resource: Optional[str] = None,
) -> torchx.AppDef:
    """Runs the model interpretability app on the model outputted by the training
    component.

    Args:
        image: image to run (e.g. foobar:latest)
        load_path: path to load pretrained model from
        data_path: path to the data to load
        output_path: output path for model checkpoints (e.g. file:///foo/bar)
        resource: the resources to use
    """
    return torchx.AppDef(
        name="cv-interpret",
        roles=[
            torchx.Role(
                name="worker",
                entrypoint="python",
                args=[
                    "-m",
                    "torchx.examples.apps.lightning_classy_vision.interpret",
                    "--load_path",
                    load_path,
                    "--data_path",
                    data_path,
                    "--output_path",
                    output_path,
                ],
                image=image,
                resource=named_resources[resource]
                if resource
                else torchx.Resource(cpu=1, gpu=0, memMB=1024),
            )
        ],
    )
