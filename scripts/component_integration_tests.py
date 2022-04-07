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

import example_app_defs as examples_app_defs_providers
import torchx.components.integration_tests.component_provider as component_provider
from integ_test_utils import BuildInfo, MissingEnvError, build_images, push_images
from torchx.components.integration_tests.integ_tests import (
    IntegComponentTest,
    SchedulerInfo,
)
from torchx.schedulers import get_scheduler_factories


# pyre-ignore-all-errors[21] # Cannot find module utils
# pyre-ignore-all-errors[11]


def build_and_push_image() -> BuildInfo:
    build = build_images()
    push_images(build)
    return build


def get_k8s_sched_info(image: str) -> SchedulerInfo:
    cfg = {
        "namespace": "torchx-dev",
        "queue": "default",
    }
    return SchedulerInfo(name="kubernetes", image=image, cfg=cfg)


def get_ray_sched_info(image: str) -> SchedulerInfo:
    cfg = {
        "namespace": "torchx-dev",
    }
    return SchedulerInfo(name="ray", image=image, cfg=cfg)


def argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process some integers.")
    choices = list(get_scheduler_factories().keys())
    parser.add_argument("--scheduler", required=True, choices=choices)
    return parser


def main() -> None:
    args = argparser().parse_args()
    scheduler = args.scheduler

    print("Starting components integration tests")
    torchx_image = "dummy_image"
    dryrun: bool = False
    if scheduler in ("kubernetes", "local_docker", "aws_batch"):
        try:
            build = build_and_push_image()
            torchx_image = build.torchx_image
        except MissingEnvError:
            dryrun = True
            print("Skip running tests, executed only docker build step")
    test_suite: IntegComponentTest = IntegComponentTest(timeout=15 * 60)  # 15 minutes

    def run_components(info: SchedulerInfo) -> None:
        test_suite.run_components(
            component_provider,
            scheduler_infos=[info],
            dryrun=dryrun,
        )

    def run_examples(info: SchedulerInfo) -> None:
        test_suite.run_components(
            examples_app_defs_providers,
            scheduler_infos=[info],
            dryrun=dryrun,
        )

    if scheduler == "kubernetes":
        info = get_k8s_sched_info(torchx_image)
        run_components(info)
        run_examples(info)
    elif scheduler == "local_cwd":
        info = SchedulerInfo(name=scheduler, image=os.getcwd())
        run_components(info)
    elif scheduler == "local_docker":
        info = SchedulerInfo(name=scheduler, image=torchx_image)
        run_components(info)
        run_examples(info)
    elif scheduler == "aws_batch":
        info = SchedulerInfo(
            name=scheduler,
            image=torchx_image,
            cfg={
                "queue": "torchx",
            },
        )
        run_components(info)
    elif scheduler == "ray":
        info = get_ray_sched_info(torchx_image)
        run_components(info)
    else:
        raise ValueError(f"component tests missing support for {scheduler}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(e)
