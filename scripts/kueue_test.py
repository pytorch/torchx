#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse

from integ_test_utils import build_images, BuildInfo, MissingEnvError, push_images
from torchx.components.dist import ddp
from torchx.runner import get_runner
from torchx.specs import AppState
from torchx.util.types import none_throws


def argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Kueue dist trainer integration test runner."
    )
    parser.add_argument("--container_repo", type=str)
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Does not actually submit the app," " just prints the scheduler request",
    )
    return parser


def build_and_push_image(container_repo: str) -> BuildInfo:
    build = build_images()
    push_images(build, container_repo=container_repo)
    return build


def run_kueue_test(dryrun: bool = False):
    # Gather args & build image
    print("Building image")
    args = argparser().parse_args()
    build = build_and_push_image(args.container_repo)
    image = build.torchx_image
    # Create the app definition
    runner = get_runner("kueue")
    app = ddp(
        name="kueue-test",
        image=image,
        m="torchx.examples.apps.lightning.train",
        cpu=1,
        memMB=4000,
        j="1x1",
    )
    # Pass config variables
    cfg = {"namespace": "torchx-dev", "local_queue": "torchx-local-queue"}
    print("Submitting job")
    if dryrun:
        dryrun_info = runner.dryrun(app, "kueue", cfg)
        print(f"Dryrun info: {dryrun_info}")
    else:
        app_handle = runner.run(app, "kueue", cfg)
        print(app_handle)
        runner.wait(app_handle)
        final_status = runner.status(app_handle)
        print(f"Final status: {final_status}")
        if none_throws(final_status).state != AppState.SUCCEEDED:
            raise Exception(f"Dist app failed with status: {final_status}")


def main() -> None:
    args = argparser().parse_args()

    try:
        run_kueue_test(args.dryrun)
    except MissingEnvError:
        print("Skip runnig tests, executed only docker build step")


if __name__ == "__main__":
    main()
