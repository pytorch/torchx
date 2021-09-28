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

# pyre-ignore-all-errors[21] # Cannot find module utils
# pyre-ignore-all-errors[11]

import example_app_defs as examples_app_defs_providers
import torchx.components.integration_tests.component_provider as component_provider
from integ_test_utils import (
    MissingEnvError,
    build_images,
    push_images,
    BuildInfo,
)
from torchx.components.integration_tests.integ_tests import (
    IntegComponentTest,
    SchedulerInfo,
)
from torchx.specs import RunConfig


def build_and_push_image() -> BuildInfo:
    build = build_images()
    push_images(build)
    return build


def get_k8s_sched_info(image: str) -> SchedulerInfo:
    cfg = RunConfig()
    cfg.set("namespace", "torchx-dev")
    cfg.set("queue", "default")
    return SchedulerInfo(name="kubernetes", image=image, runconfig=cfg)


def get_local_cwd_sched_info(image: str) -> SchedulerInfo:
    return SchedulerInfo(name="local_cwd", image=image, runconfig=RunConfig())


def get_local_docker_sched_info(image: str) -> SchedulerInfo:
    return SchedulerInfo(name="local_docker", image=image, runconfig=RunConfig())


def main() -> None:
    parser = argparse.ArgumentParser(description="kubernetes integration test runner")
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Does not actually submit the app," " just prints the scheduler request",
    )
    args = parser.parse_args()
    print("Starting components integration tests")
    torchx_image = "dummy_image"
    examples_image = "dummy_image"
    try:
        build = build_and_push_image()
        torchx_image = build.torchx_image
        examples_image = build.examples_image
    except MissingEnvError:
        print("Skip runnig tests, executed only docker buid step")
    print("aivanou test run with: ", args.dryrun, torchx_image, examples_image)
    test_suite = IntegComponentTest(timeout=900)  # 15 minutes
    test_suite.run_components(
        component_provider,
        scheduler_infos=[
            get_local_cwd_sched_info(os.getcwd()),
            get_local_docker_sched_info(torchx_image),
            get_k8s_sched_info(torchx_image),
        ],
        dryrun=args.dryrun,
    )

    test_suite.run_components(
        examples_app_defs_providers,
        scheduler_infos=[
            get_local_docker_sched_info(examples_image),
            get_k8s_sched_info(examples_image),
        ],
        dryrun=args.dryrun,
    )


if __name__ == "__main__":
    main()
