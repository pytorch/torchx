#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Kubernetes integration tests.
"""

import binascii
import os
import subprocess
import tempfile
from torchx.runner import get_runner
from torchx.specs import Resource, named_resources, RunConfig, AppState
from examples.apps.dist_cifar.component import trainer


class MissingEnvError(AssertionError):
    pass


def rand_id() -> str:
    return binascii.b2a_hex(os.urandom(8)).decode("utf-8")


def getenv_asserts(env: str) -> str:
    v = os.getenv(env)
    if not v:
        raise MissingEnvError(f"must have {env} environment variable")
    return v


def run(*args: str) -> None:
    print(f"run {args}")
    subprocess.run(args, check=True)


def build_examples_canary(id: str) -> str:
    examples_tag = "torchx_examples_canary"

    print(f"building {examples_tag}")
    run("docker", "build", "-t", examples_tag, "examples/apps/")

    return examples_tag


GiB: int = 1024


def gpu_resource() -> Resource:
    return Resource(
        cpu=2,
        gpu=1,
        memMB=16 * GiB,
    )


def register_gpu_resource() -> None:
    res = gpu_resource()
    print(f"Registering resource: {res}")
    named_resources["GPU_X1"] = res


def get_container_tag(id: str) -> str:
    CONTAINER_REPO = getenv_asserts("CONTAINER_REPO")
    return f"{CONTAINER_REPO}:canary_{id}_examples"


def build_and_push_image():
    print("Building docker image")
    img_id = build_examples_canary(rand_id())
    img_tag = get_container_tag(img_id)
    run("docker", "tag", img_id, img_tag)
    run("docker", "push", img_tag)
    print(f"Pushing docker image: {img_tag}")
    return img_tag

def get_dryrun_option()->bool:
    return os.environ.get("EXEC_MODE", "run") =="dryrun"

def run_job(tmp_dir: str) -> None:
    register_gpu_resource()
    image = build_and_push_image()
    runner = get_runner("kubeflow-dist-runner")
    args = ("--output_path", tmp_dir)
    train_app = trainer(
        *args,
        image=image,
        resource="GPU_X1",
        nnodes=2,
        rdzv_backend="etcd-v2",
        rdzv_endpoint="etcd-server:2379"
    )
    cfg = RunConfig()
    cfg.set("namespace", "default")
    cfg.set("queue", "default")
    print("Submitting pods")
    dryrun = get_dryrun_option()
    app_handle = runner.run(train_app, "kubernetes", cfg,dryrun)
    print(app_handle)
    runner.wait(app_handle)
    final_status = runner.status(app_handle)
    print(f"Final status: {final_status}")
    if final_status.state != AppState.SUCCEEDED:
        raise Exception(f"Dist app failed with status: {final_status}")


def main() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        run_job(tmpdir)


if __name__ == "__main__":
    main()
