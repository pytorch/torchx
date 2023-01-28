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
import logging
import os

import example_app_defs as examples_app_defs_providers
import torchx.components.integration_tests.component_provider as component_provider
from integ_test_utils import build_images, BuildInfo, MissingEnvError, push_images
from torchx.cli.colors import BLUE, ENDC, GRAY
from torchx.components.integration_tests.integ_tests import IntegComponentTest
from torchx.schedulers import get_scheduler_factories


logging.basicConfig(
    level=logging.INFO,
    format=f"{GRAY}%(asctime)s{ENDC} {BLUE}%(name)-12s{ENDC} %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# pyre-ignore-all-errors[21] # Cannot find module utils
# pyre-ignore-all-errors[11]


def build_and_push_image() -> BuildInfo:
    build = build_images()
    push_images(build)
    return build


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
    dryrun = False

    if scheduler in (
        "kubernetes",
        "kubernetes_mcad",
        "local_docker",
        "aws_batch",
        "lsf",
        "gcp_batch",
    ):
        try:
            build = build_and_push_image()
            torchx_image = build.torchx_image
        except MissingEnvError:
            dryrun = True
            print("Skip running tests, executed only docker build step")

    run_parameters = {
        "kubernetes": {
            "providers": [
                component_provider,
                examples_app_defs_providers,
            ],
            "image": torchx_image,
            "cfg": {
                "namespace": "torchx-dev",
                "queue": "default",
            },
        },
        "kubernetes_mcad": {
            "providers": [
                component_provider,
                examples_app_defs_providers,
            ],
            "image": torchx_image,
            "cfg": {
                "namespace": "torchx-dev",
            },
        },
        "local_cwd": {
            "providers": [
                component_provider,
            ],
            "image": os.getcwd(),
            "cfg": {},
        },
        "local_docker": {
            "providers": [
                component_provider,
                examples_app_defs_providers,
            ],
            "image": torchx_image,
            "cfg": {},
        },
        "aws_batch": {
            "providers": [
                component_provider,
            ],
            "image": torchx_image,
            "cfg": {
                "queue": "torchx",
            },
        },
        "gcp_batch": {
            "providers": [
                component_provider,
            ],
            "image": torchx_image,
            "cfg": {},
        },
        "ray": {
            "providers": [
                component_provider,
            ],
            "image": torchx_image,
            "cfg": {
                "requirements": "",
            },
            "workspace": f"file://{os.getcwd()}",
        },
        "lsf": {
            "providers": [
                component_provider,
            ],
            "image": torchx_image,
            "cfg": {
                "runtime": "docker",
                "jobdir": "/mnt/data/torchx",
                "host_network": True,
            },
        },
    }

    params = run_parameters[scheduler]
    test_suite: IntegComponentTest = IntegComponentTest()
    for provider in params["providers"]:
        test_suite.run_components(
            module=provider,
            scheduler=scheduler,
            image=params["image"],
            cfg=params["cfg"],
            dryrun=dryrun,
            workspace=params.get("workspace"),
        )


if __name__ == "__main__":
    main()
