#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Kubernetes integration tests.
"""
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
    print("Starting components integration tests")
    torchx_image = "dummy_image"
    dryrun: bool = False
    try:
        build = build_and_push_image()
        torchx_image = build.torchx_image
    except MissingEnvError:
        dryrun = True
        print("Skip runnig tests, executed only docker buid step")
    test_suite = IntegComponentTest(timeout=600)  # 10 minutes
    test_suite.run_components(
        component_provider,
        scheduler_infos=[
            get_local_cwd_sched_info(os.getcwd()),
            get_k8s_sched_info(torchx_image),
        ],
        dryrun=dryrun,
    )

    # Run components on `local_docker`  scheduler in sequence due to
    # docker APIs are not atomic. Some of the APIs, e.g. `create_network`
    # cause a race condition, making several networks with the same name to be created.
    test_suite.run_components(
        component_provider,
        scheduler_infos=[
            get_local_docker_sched_info(torchx_image),
        ],
        dryrun=dryrun,
        run_in_parallel=False,
    )

    test_suite.run_components(
        examples_app_defs_providers,
        scheduler_infos=[
            get_local_docker_sched_info(torchx_image),
            get_k8s_sched_info(torchx_image),
        ],
        dryrun=dryrun,
    )


if __name__ == "__main__":
    main()
