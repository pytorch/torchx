# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import fsspec
from torchx.specs import Role
from torchx.workspace.docker_workspace import DockerWorkspace


def has_docker() -> bool:
    try:
        import docker

        docker.from_env()
        return True
    except (ImportError, docker.errors.DockerException):
        return False


if has_docker():

    class DockerWorkspaceTest(unittest.TestCase):
        def test_docker_workspace(self) -> None:
            fs = fsspec.filesystem("memory")
            fs.mkdirs("test_workspace/bar", exist_ok=True)
            with fs.open("test_workspace/bar/foo.sh", "w") as f:
                f.write("exit 0")

            role = Role(
                name="ping",
                image="busybox",
                entrypoint="sh",
                args=["bar/foo.sh"],
            )

            workspace = DockerWorkspace()
            workspace.build_workspace_and_update_role(role, "memory://test_workspace")

            self.assertNotEqual("busybox", role.image)
