#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""


"""

from torchx.runner import get_runner
from torchx.specs import Resource, named_resources, RunConfig
from examples.apps.dist_cifar.component import trainer

GiB: int = 1024


def p3_2xlarge() -> Resource:
    return Resource(
        # TODO(aivanou): Determine why the number of CPUs allowed to
        # be requested via volcano is N-1
        cpu=7,
        gpu=1,
        memMB=61 * GiB,
    )


def register_gpu_resource() -> None:
    res = p3_2xlarge()
    print(f"Registering resource: {res}")
    named_resources["p3_2xlarge"] = res


if __name__ == "__main__":
    register_gpu_resource()
    runner = get_runner("kubernetes")
    args = ("--output_path", "/tmp")
    img = "public.ecr.aws/y5x3w0a7/aivanou-tests"
    train_app = trainer(
        *args,
        image=img,
        resource="p3_2xlarge",
        nnodes=2,
        rdzv_backend="etcd-v2",
        rdzv_endpoint="etcd-server:2379"
    )
    cfg = RunConfig()
    cfg.set("namespace", "default")
    cfg.set("queue", "test")
    app_handle = runner.run(train_app, cfg)
    print(app_handle)
    runner.wait(app_handle)
