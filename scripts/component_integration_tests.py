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

# pyre-ignore-all-errors[21] # Cannot find module utils
# pyre-ignore-all-errors[11]
from integ_test_utils import (
    MissingEnvError,
    build_images,
    push_images,
    BuildInfo,
)
from torchx.components.integration_tests.integ_tests import IntegComponentTest


def build_and_push_image() -> BuildInfo:
    build = build_images()
    push_images(build)
    return build


def main() -> None:
    parser = argparse.ArgumentParser(description="kubernetes integration test runner")
    parser.add_argument(
        "--dryrun",
        action="store_true",
        help="Does not actually submit the app," " just prints the scheduler request",
    )
    args = parser.parse_args()

    try:
        build = build_and_push_image()
        test_suite = IntegComponentTest()
        test_suite.run_builtin_components(
            image=build.examples_image,
            schedulers=["local", "kubernetes"],
            dryrun=args.dryrun,
        )
    except MissingEnvError:
        print("Skip runnig tests, executed only docker buid step")


if __name__ == "__main__":
    main()
