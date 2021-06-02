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
from typing import Optional

import kfp
import requests


NAMESPACE: Optional[str] = os.getenv("KFP_NAMESPACE")
assert NAMESPACE, "missing KFP_NAMESPACE environment variable"


def get_client() -> kfp.Client:
    HOST = os.getenv("KFP_HOST")
    assert HOST, "must have KFP_HOST environment variable"
    USERNAME = os.getenv("KFP_USERNAME")
    assert USERNAME, "must have KFP_USERNAME environment variable"
    PASSWORD = os.getenv("KFP_PASSWORD")
    assert PASSWORD, "must have KFP_PASSWORD environment variable"

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


def build_canary(id: str) -> str:
    CONTAINER_REPO = os.getenv("EXAMPLES_CONTAINER_REPO")
    assert CONTAINER_REPO, "must have EXAMPLES_CONTAINER_REPO environment variable"
    tag = f"{CONTAINER_REPO}:canary_{id}"
    print(f"building {tag}")
    run("docker", "build", "-t", tag, "examples/")
    run("docker", "push", tag)
    return tag


def run_test() -> None:
    STORAGE_PATH = os.getenv("INTEGRATION_TEST_STORAGE")
    assert STORAGE_PATH, "must have INTEGRATION_TEST_STORAGE environment variable"
    id = f"{getuser()}_{rand_id()}"
    image = build_canary(id)
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
            image,
            "--package_path",
            package,
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
        result = resp.wait_for_run_completion(timeout=1 * 60 * 60)  # 1 hour
        print("finished", result)
        assert result.run.status == "Succeeded", "run didn't succeed"


if __name__ == "__main__":
    run_test()
