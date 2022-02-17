# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import MagicMock

import fsspec
from torchx.specs import Role, AppDef
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


class DockerWorkspaceMockTest(unittest.TestCase):
    def test_update_app_images(self) -> None:
        app = AppDef(
            name="foo",
            roles=[
                Role(
                    name="a",
                    image="sha256:hasha",
                ),
                Role(name="b", image="sha256:hashb"),
                Role(
                    name="c",
                    image="c",
                ),
            ],
        )
        want = AppDef(
            name="foo",
            roles=[
                Role(
                    name="a",
                    image="example.com/repo:hasha",
                ),
                Role(
                    name="b",
                    image="example.com/repo:hashb",
                ),
                Role(
                    name="c",
                    image="c",
                ),
            ],
        )
        # no image_repo
        with self.assertRaisesRegex(KeyError, "image_repo"):
            DockerWorkspace()._update_app_images(app, {})
        # with image_repo
        images_to_push = DockerWorkspace()._update_app_images(
            app,
            {
                "image_repo": "example.com/repo",
            },
        )
        self.assertEqual(
            images_to_push,
            {
                "sha256:hasha": ("example.com/repo", "hasha"),
                "sha256:hashb": ("example.com/repo", "hashb"),
            },
        )
        self.assertEqual(app, want)

    def test_push_images(self) -> None:
        client = MagicMock()
        img = MagicMock()
        client.images.get.return_value = img
        workspace = DockerWorkspace(docker_client=client)
        workspace._push_images(
            {
                "sha256:hasha": ("example.com/repo", "hasha"),
                "sha256:hashb": ("example.com/repo", "hashb"),
            }
        )
        self.assertEqual(client.images.get.call_count, 2)
        self.assertEqual(img.tag.call_count, 2)
        self.assertEqual(client.images.push.call_count, 2)

    def test_push_images_empty(self) -> None:
        workspace = DockerWorkspace()
        workspace._push_images({})
