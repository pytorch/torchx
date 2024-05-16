#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
Kubernetes integration tests.
"""

import logging

from integ_test_utils import (
    build_images,
    BuildInfo,
    getenv_asserts,
    MissingEnvError,
    push_images,
)
from torchx.components.dist import ddp as dist_ddp
from torchx.runner import get_runner
from torchx.specs import _named_resource_factories, AppState, Resource
from torchx.util.types import none_throws
from component_integration_tests import build_and_push_image

log: logging.Logger = logging.getLogger(__name__)

GiB: int = 1024


def register_gpu_resource() -> None:
    res = Resource(
        cpu=2,
        gpu=1,
        memMB=8 * GiB,
    )
    print(f"Registering resource: {res}")
    _named_resource_factories["GPU_X1"] = lambda: res


# def build_and_push_image() -> BuildInfo:
#     build = build_images()
#     push_images(build, container_repo=getenv_asserts("CONTAINER_REPO"))
#     return build


def run_job() -> None:
    register_gpu_resource()
    build = build_and_push_image(container_repo=getenv_asserts("CONTAINER_REPO"))
    image = build.torchx_image
    runner = get_runner()
    train_app = dist_ddp(
        m="torchx.examples.apps.compute_world_size.main",
        name="ddp-trainer",
        image=image,
        cpu=1,
        j="2x2",
        max_retries=3,
        env={
            "LOGLEVEL": "INFO",
        },
    )
    cfg = {
        "namespace": "torchx-dev",
        "queue": "default",
    }
    app_handle = runner.run(train_app, "kubernetes", cfg)
    runner.wait(app_handle)
    final_status = runner.status(app_handle)
    print(f"Final status: {final_status}")
    if none_throws(final_status).state != AppState.SUCCEEDED:
        raise Exception(f"Dist app failed with status: {final_status}")


def main() -> None:
    try:
        run_job()
    except MissingEnvError:
        print("Skip runnig tests, executed only docker buid step")


if __name__ == "__main__":
    main()
