#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import binascii
import os
import os.path
import subprocess
import tempfile
from getpass import getuser

import kfp
import requests


def getenv_asserts(env: str) -> str:
    v = os.getenv(env)
    assert v, f"must have {env} environment variable"
    return v


NAMESPACE: str = getenv_asserts("KFP_NAMESPACE")
HOST: str = getenv_asserts("KFP_HOST")


def get_client() -> kfp.Client:
    USERNAME = getenv_asserts("KFP_USERNAME")
    PASSWORD = getenv_asserts("KFP_PASSWORD")

    session = requests.Session()
    response = session.get(HOST)

    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = {"login": USERNAME, "password": PASSWORD}
    session.post(response.url, headers=headers, data=data)
    session_cookie = session.cookies.get_dict()["authservice_session"]

    return kfp.Client(
        host=f"{HOST}/pipeline",
        cookies=f"authservice_session={session_cookie}",
        namespace=NAMESPACE,
    )


def run(*args: str) -> None:
    subprocess.run(args, check=True)


def rand_id() -> str:
    return binascii.b2a_hex(os.urandom(8)).decode("utf-8")


def build_examples_canary(id: str) -> str:
    EXAMPLES_CONTAINER_REPO = getenv_asserts("EXAMPLES_CONTAINER_REPO")
    examples_tag = f"{EXAMPLES_CONTAINER_REPO}:canary_{id}"

    print(f"building {examples_tag}")
    run("docker", "build", "-t", examples_tag, "examples/")
    run("docker", "push", examples_tag)

    return examples_tag


def build_torchx_canary(id: str) -> str:
    TORCHX_CONTAINER_REPO = getenv_asserts("TORCHX_CONTAINER_REPO")
    torchx_tag = f"{TORCHX_CONTAINER_REPO}:canary_{id}"

    print(f"building {torchx_tag}")
    run("./torchx/runtime/container/build.sh")
    run("docker", "tag", "torchx", torchx_tag)
    run("docker", "push", torchx_tag)

    return torchx_tag


def run_test() -> None:
    STORAGE_PATH = os.getenv("INTEGRATION_TEST_STORAGE")
    assert STORAGE_PATH, "must have INTEGRATION_TEST_STORAGE environment variable"
    id = f"{getuser()}_{rand_id()}"
    examples_image = build_examples_canary(id)
    torchx_image = build_torchx_canary(id)
    root = os.path.join(STORAGE_PATH, id)
    data = os.path.join(root, "data")
    output = os.path.join(root, "output")
    with tempfile.TemporaryDirectory() as tmpdir:
        print("generating pipeline spec")
        package = os.path.join(tmpdir, "pipeline.yml")
        run(
            "examples/kfp_pipeline.py",
            "--data_path",
            data,
            "--output_path",
            output,
            "--image",
            examples_image,
            "--package_path",
            package,
            "--torchx_image",
            torchx_image,
            "--model_name",
            f"tiny_image_net_{id}",
        )

        print("launching run")
        client = get_client()
        resp = client.create_run_from_pipeline_package(
            package,
            arguments={},
            namespace=NAMESPACE,
            experiment_name="integration-tests",
            run_name=f"integration test {id}",
        )
        print("waiting for completion", resp)
        ui_url = f"{HOST}/_/pipeline/#/runs/details/{resp.run_id}"
        print(f"view run at {ui_url}")
        result = resp.wait_for_run_completion(timeout=1 * 60 * 60)  # 1 hour
        print("finished", result)
        print(f"view run at {ui_url}")
        assert result.run.status == "Succeeded", "run didn't succeed"


if __name__ == "__main__":
    run_test()
