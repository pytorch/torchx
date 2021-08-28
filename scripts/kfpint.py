#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This file runs the KFP integration tests on KFP cluster. There's a number of
environment variables that need to be setup as well as the cluster.

See examples/pipelines/kfp/ for more information on how the cluster is used.

Cluster setup:

You'll need a KubeFlow Pipelines cluster with a torchserve instance with svc
name torchserve on the default namespace.

* https://www.kubeflow.org/docs/started/installing-kubeflow/
* https://github.com/pytorch/serve/blob/master/kubernetes/README.md

Environment variables:

```
export KFP_HOST=<kfp HTTP URL without any path>
export KFP_USERNAME=<kfp username>
export KFP_PASSWORD=<kfp password>
export KFP_NAMESPACE=<kfp namespace>
export INTEGRATION_TEST_STORAGE=<cloud storage path>
export EXAMPLES_CONTAINER_REPO=<docker repo>
export TORCHX_CONTAINER_REPO=<docker repo>
```

Once you have everything setup you can just run:

scripts/kfpint.py


"""

import argparse
import asyncio
import binascii
import dataclasses
import json
import os
import os.path
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from getpass import getuser
from typing import Optional, Iterator, Any, TypeVar, Callable

import kfp

T = TypeVar("T")


def retryer(f: Callable[..., T]) -> Callable[..., T]:
    retries: int = 5
    backoff: int = 3

    def wrapper(*args, **kwargs):
        curr_retries = 0
        while True:
            try:
                return f(*args, **kwargs)
            except:
                if curr_retries == retries:
                    raise
                else:
                    sleep = backoff * 2 ** curr_retries
                    fn_name = f.__qualname__
                    print(f"retrying `{fn_name}` request after {sleep} seconds")
                    time.sleep(sleep)
                    curr_retries += 1
                    continue

    return wrapper


@dataclasses.dataclass
class BuildInfo:
    id: str
    torchx_image: str
    examples_image: str


class MissingEnvError(AssertionError):
    pass


def getenv_asserts(env: str) -> str:
    v = os.getenv(env)
    if not v:
        raise MissingEnvError(f"must have {env} environment variable")
    return v


def run_in_bg(*args: str) -> subprocess.Popen:
    print(f"run {args}")
    return subprocess.Popen(args)


def get_free_port() -> int:
    return 8080


def enable_port_forward() -> subprocess.Popen:
    # Enable port forward via running background process.
    # Kubernetes python does not support a clean way of
    # Kubernetes python cli provides a socket, more info:
    # https://github.com/kubernetes-client/python/blob/master/examples/pod_portforward.py
    # The drawback of this method is that we have to monkey patch
    # the urllib, which is used by the kfp client.
    # This approach is more cleaner than to use the python cli directly.
    namespace = getenv_asserts("KFP_NAMESPACE")
    local_port = get_free_port()
    return run_in_bg(
        "kubectl",
        "port-forward",
        "-n",
        namespace,
        "svc/ml-pipeline-ui",
        f"{local_port}:80",
    )


@retryer
def get_client(host: str) -> kfp.Client:
    return kfp.Client(host=f"{host}/pipeline")


def run(*args: str) -> None:
    print(f"run {args}")
    subprocess.run(args, check=True)


def rand_id() -> str:
    return binascii.b2a_hex(os.urandom(8)).decode("utf-8")


def torchx_container_tag(id: str) -> str:
    CONTAINER_REPO = getenv_asserts("CONTAINER_REPO")
    return f"{CONTAINER_REPO}:canary_{id}_torchx"


def examples_container_tag(id: str) -> str:
    CONTAINER_REPO = getenv_asserts("CONTAINER_REPO")
    return f"{CONTAINER_REPO}:canary_{id}_examples"


def build_examples_canary(id: str) -> str:
    examples_tag = "torchx_examples_canary"

    print(f"building {examples_tag}")
    run("docker", "build", "-t", examples_tag, "examples/apps/")

    return examples_tag


def build_torchx_canary(id: str) -> str:
    torchx_tag = "torchx_canary"

    print(f"building {torchx_tag}")
    run("./torchx/runtime/container/build.sh")
    run("docker", "tag", "torchx", torchx_tag)

    return torchx_tag


def build_images() -> BuildInfo:
    id = f"{getuser()}_{rand_id()}"
    examples_image = build_examples_canary(id)
    torchx_image = build_torchx_canary(id)
    return BuildInfo(
        id=id,
        torchx_image=torchx_image,
        examples_image=examples_image,
    )


META_FILE = "meta"
IMAGES_FILE = "images.tar.zst"


def save_build(path: str, build: BuildInfo) -> None:
    meta_path = os.path.join(path, META_FILE)
    with open(meta_path, "wt") as f:
        json.dump(dataclasses.asdict(build), f)


def push_images(build: BuildInfo) -> None:
    examples_tag = examples_container_tag(build.id)
    run("docker", "tag", build.examples_image, examples_tag)
    build.examples_image = examples_tag

    torchx_tag = torchx_container_tag(build.id)
    run("docker", "tag", build.torchx_image, torchx_tag)
    build.torchx_image = torchx_tag

    run("docker", "push", examples_tag)
    run("docker", "push", torchx_tag)


def save_advanced_pipeline_spec(path: str, build: BuildInfo) -> None:
    print("generating advanced_pipeline spec")

    id = build.id
    torchx_image = build.torchx_image
    examples_image = build.examples_image

    STORAGE_PATH = os.getenv("INTEGRATION_TEST_STORAGE", "/tmp/storage")
    root = os.path.join(STORAGE_PATH, id)
    data = os.path.join(root, "data")
    output = os.path.join(root, "output")
    logs = os.path.join(root, "logs")

    save_pipeline_spec(
        path,
        "advanced_pipeline.py",
        "--data_path",
        data,
        "--output_path",
        output,
        "--image",
        examples_image,
        "--log_path",
        logs,
        "--torchx_image",
        torchx_image,
        "--model_name",
        f"tiny_image_net_{id}",
    )


def save_pipeline_spec(path: str, pipeline_file: str, *args: str) -> None:
    print(f"generating pipeline spec for {pipeline_file}")

    with tempfile.TemporaryDirectory() as tmpdir:
        run(os.path.join("examples/pipelines/kfp", pipeline_file), *args)
        shutil.copy("pipeline.yaml", path)


@contextmanager
def path_or_tmp(path: Optional[str]) -> Iterator[str]:
    if path:
        os.makedirs(path, exist_ok=True)
        yield path
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir


def run_pipeline(build: BuildInfo, pipeline_file: str) -> object:
    print(f"launching pipeline {pipeline_file}")
    HOST: str = getenv_asserts("KFP_HOST")
    HOST: str = "http://localhost:8080"

    client = get_client(HOST)
    resp = client.create_run_from_pipeline_package(
        pipeline_file,
        arguments={},
        experiment_name="integration-tests",
        run_name=f"integration test {build.id} - {os.path.basename(pipeline_file)}",
    )
    ui_url = f"{HOST}/_/pipeline/#/runs/details/{resp.run_id}"
    print(f"{resp.run_id} - launched! view run at {ui_url}")
    return resp


def wait_for_pipeline(
        resp: Any,  # pyre-fixme: KFP doesn't have a response type
) -> None:
    print(f"{resp.run_id} - waiting for completion")
    result = resp.wait_for_run_completion(
        timeout=1 * 60 * 60,
    )  # 1 hour
    print(f"{resp.run_id} - finished: {result}")
    assert result.run.status == "Succeeded", "run didn't succeed"


async def main() -> None:
    parser = argparse.ArgumentParser(description="kfp integration test runner")
    parser.add_argument(
        "--path",
        type=str,
        help="path to place the files",
    )
    parser.add_argument(
        "--load",
        help="if specified load the build from path instead of building",
        action="store_true",
    )
    parser.add_argument(
        "--save",
        help="if specified save the build to path and exit",
        action="store_true",
    )
    args = parser.parse_args()

    with path_or_tmp(args.path) as path:
        advanced_pipeline_file = os.path.join(path, "advanced_pipeline.yaml")
        intro_pipeline_file = os.path.join(path, "intro_pipeline.yaml")
        dist_pipeline_file = os.path.join(path, "dist_pipeline.yaml")
        build = build_images()
        try:
            push_images(build)
        except MissingEnvError as e:
            print(f"Missing environments, only building: {e}")
            return
        finally:
            save_advanced_pipeline_spec(advanced_pipeline_file, build)
            save_pipeline_spec(intro_pipeline_file, "intro_pipeline.py")
            save_pipeline_spec(dist_pipeline_file, "dist_pipeline.py")

        pipeline_files = [
            advanced_pipeline_file,
            intro_pipeline_file,
            dist_pipeline_file,
        ]
        runs = [run_pipeline(build, pipeline_file) for pipeline_file in pipeline_files]
        for run in runs:
            wait_for_pipeline(run)


import time

if __name__ == "__main__":
    port_forward_proc = enable_port_forward()
    try:
        asyncio.run(main())
    finally:
        port_forward_proc.kill()
