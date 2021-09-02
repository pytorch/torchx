#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import binascii
import dataclasses
import os
import subprocess
from getpass import getuser


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


def rand_id() -> str:
    return binascii.b2a_hex(os.urandom(8)).decode("utf-8")


def run(*args: str) -> None:
    print(f"run {args}")
    subprocess.run(args, check=True)


def run_in_bg(*args: str) -> "subprocess.Popen[str]":
    print(f"run {args}")
    return subprocess.Popen(args)


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


def torchx_container_tag(id: str) -> str:
    CONTAINER_REPO = getenv_asserts("CONTAINER_REPO")
    return f"{CONTAINER_REPO}:canary_{id}_torchx"


def examples_container_tag(id: str) -> str:
    CONTAINER_REPO = getenv_asserts("CONTAINER_REPO")
    return f"{CONTAINER_REPO}:canary_{id}_examples"


def build_images() -> BuildInfo:
    id = f"{getuser()}_{rand_id()}"
    examples_image = build_examples_canary(id)
    torchx_image = build_torchx_canary(id)
    return BuildInfo(
        id=id,
        torchx_image=torchx_image,
        examples_image=examples_image,
    )


def push_images(build: BuildInfo) -> None:
    examples_tag = examples_container_tag(build.id)
    run("docker", "tag", build.examples_image, examples_tag)
    build.examples_image = examples_tag

    torchx_tag = torchx_container_tag(build.id)
    run("docker", "tag", build.torchx_image, torchx_tag)
    build.torchx_image = torchx_tag

    run("docker", "push", examples_tag)
    run("docker", "push", torchx_tag)
