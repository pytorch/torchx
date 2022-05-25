#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Kubernetes integration tests.
"""

import argparse
import os

from integ_test_utils import build_images, BuildInfo, MissingEnvError, push_images
from pyre_extensions import none_throws
from torchx.components.dist import ddp as dist_ddp
from torchx.runner import get_runner
from torchx.specs import AppState, named_resources, Resource


# pyre-ignore-all-errors[21] # Cannot find module utils
# pyre-ignore-all-errors[11]


GiB: int = 1024


# TODO(aivanou): remove this when partial resources are introduced.
def register_gpu_resource() -> None:
    res = Resource(
        cpu=2,
        gpu=1,
        memMB=8 * GiB,
        capabilities={
            "node.kubernetes.io/instance-type": "p3.2xlarge",
        },
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
    image = build.torchx_image
    runner = get_runner("kubeflow-dist-runner")

    storage_path = os.getenv("INTEGRATION_TEST_STORAGE", "/tmp/storage")
    root = os.path.join(storage_path, build.id)
    output_path = os.path.join(root, "output")

    args = ("--output_path", output_path)
    train_app = dist_ddp(
        *("--output_path", output_path),
        m="torchx.examples.apps.lightning_classy_vision.train",
        image=image,
        h="GPU_X1",
        j="2x1",
    )
    print(f"Starting Trainer with args: {args}")
    cfg = {
        "namespace": "default",
        "queue": "default",
    }
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
