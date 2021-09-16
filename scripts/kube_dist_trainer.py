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
import tempfile

import examples.apps.dist_cifar.component as dist_cifar_component
import examples.apps.lightning_classy_vision.component as cv_component

# pyre-ignore-all-errors[21] # Cannot find module utils
# pyre-ignore-all-errors[11]
from integ_test_utils import (
    MissingEnvError,
    build_images,
    push_images,
    BuildInfo,
)
from torchx.components.component_test_base import ComponentTestCase
from torchx.specs import Resource, named_resources

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


class KuberneteschedTest(ComponentTestCase):
    def run_dist_cifar(self, build_info: BuildInfo, dryrun: bool = False) -> None:
        register_gpu_resource()
        storage_path = os.getenv("INTEGRATION_TEST_STORAGE", "/tmp/storage")
        root = os.path.join(storage_path, build_info.id)
        output_path = os.path.join(root, "output")

        image = build_info.examples_image

        args = ("--output_path", output_path)
        component_args = ["--output_path", output_path]
        component_kwargs = {
            "image": image,
            "resource": "GPU_X1",
            "nnodes": 2,
            "rdzv_backend": "etcd-v2",
            "rdzv_endpoint": "etcd-server.default.svc.cluster.local:2379",
        }
        print(f"Starting Trainer with args: {args}")
        self.run_component_on_k8s(
            dist_cifar_component.trainer, component_args, component_kwargs, dryrun
        )

    def run_lightning_cv(self, build_info: BuildInfo, dryrun: bool = False) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            image = build_info.examples_image
            component_kwargs = {
                "image": image,
                "output_path": tmpdir,
                "skip_export": True,
                "log_path": tmpdir,
            }
            self.run_component_on_k8s(
                cv_component.trainer, component_kwargs=component_kwargs, dryrun=dryrun
            )


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
        test_suite = KuberneteschedTest()
        test_suite.run_dist_cifar(build, args.dryrun)
        test_suite.run_lightning_cv(build, args.dryrun)
    except MissingEnvError:
        print("Skip runnig tests, executed only docker buid step")


if __name__ == "__main__":
    main()
