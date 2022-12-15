#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import os
import subprocess
from getpass import getuser

from torchx.schedulers.ids import random_id


@dataclasses.dataclass
class BuildInfo:
    id: str
    torchx_image: str


class MissingEnvError(AssertionError):
    pass


def getenv_asserts(env: str) -> str:
    v = os.getenv(env)
    if not v:
        raise MissingEnvError(f"must have {env} environment variable")
    return v


def run(*args: str) -> None:
    print(f"run {args}")
    subprocess.run(args, check=True)


def run_in_bg(*args: str) -> "subprocess.Popen[str]":
    print(f"run {args}")
    # pyre-fixme[7]: Expected `Popen[str]` but got `Popen[bytes]`.
    return subprocess.Popen(args)


def build_torchx_canary(id: str) -> str:
    torchx_tag = "torchx_canary"

    print(f"building {torchx_tag}")
    run("./torchx/runtime/container/build.sh")
    run("docker", "tag", "torchx", torchx_tag)

    return torchx_tag


def torchx_container_tag(id: str) -> str:
    CONTAINER_REPO = getenv_asserts("CONTAINER_REPO")
    return f"{CONTAINER_REPO}:canary_{id}_torchx"


def build_images() -> BuildInfo:
    id = f"{getuser()}_{random_id()}"
    torchx_image = build_torchx_canary(id)
    return BuildInfo(
        id=id,
        torchx_image=torchx_image,
    )


def push_images(build: BuildInfo) -> None:
    torchx_tag = torchx_container_tag(build.id)
    run("docker", "tag", build.torchx_image, torchx_tag)
    build.torchx_image = torchx_tag

    run("docker", "push", torchx_tag)
