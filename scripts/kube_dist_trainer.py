#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Kubernetes integration tests.
"""

import argparse
import os

from examples.apps.dist_cifar.component import trainer

# pyre-ignore-all-errors[21] # Cannot find module utils
# pyre-ignore-all-errors[11]
from integ_test_utils import (
    MissingEnvError,
    build_images,
    push_images,
    BuildInfo,
)
from pyre_extensions import none_throws
from torchx.runner import get_runner
from torchx.specs import Resource, named_resources, RunConfig, AppState

GiB: int = 1024


def register_gpu_resource() -> None:
    res = Resource(
        cpu=2,
        gpu=1,
        memMB=4 * GiB,
    )
    print(f"Registering resource: {res}")
    named_resources["GPU_X1"] = res


def build_and_push_image() -> BuildInfo:
    build = build_images()
    push_images(build)
    return build


def run_job(dryrun: bool = False) -> None:
    register_gpu_resource()
    build = build_and_push_image()
    image = build.examples_image
    runner = get_runner("kubeflow-dist-runner")

    storage_path = os.getenv("INTEGRATION_TEST_STORAGE", "/tmp/storage")
    root = os.path.join(storage_path, build.id)
    output_path = os.path.join(root, "output")

    args = ("--output_path", output_path)
    train_app = trainer(
        *args,
        image=image,
        resource="GPU_X1",
        nnodes=2,
        rdzv_backend="etcd-v2",
        rdzv_endpoint="etcd-server:2379",
    )
    print(f"Starting Trainer with args: {args}")
    cfg = RunConfig()
    cfg.set("namespace", "default")
    cfg.set("queue", "default")
    print("Submitting pods")
    if dryrun:
        dryrun_info = runner.dryrun(train_app, "kubernetes", cfg)
        print(f"Dryrun info: {dryrun_info}")
    else:
        app_handle = runner.run(train_app, "kubernetes", cfg)
        print(app_handle)
        runner.wait(app_handle)
        final_status = runner.status(app_handle)
        print(f"Final status: {final_status}")
        if none_throws(final_status).state != AppState.SUCCEEDED:
            raise Exception(f"Dist app failed with status: {final_status}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="kubernetes dist trainer integration test runner"
    )
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Does not actually submit the app," " just prints the scheduler request",
    )
    args = parser.parse_args()

    try:
        run_job(args.dryrun)
    except MissingEnvError:
        print("Skip runnig tests, executed only docker buid step")


if __name__ == "__main__":
    main()
