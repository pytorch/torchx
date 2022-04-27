# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tarfile
import unittest
from unittest.mock import MagicMock

import fsspec
from torchx.specs import AppDef, Role
from torchx.workspace.docker_workspace import _build_context, DockerWorkspace


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
            workspace.build_workspace_and_update_role(
                role, "memory://test_workspace", {}
            )

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

    def test_dockerignore(self) -> None:
        fs = fsspec.filesystem("memory")
        files = [
            "dockerignore/ignoredir/bar",
            "dockerignore/ignoredir/recursive/bar",
            "dockerignore/dir1/bar",
            "dockerignore/dir/ignorefileglob1",
            "dockerignore/dir/recursive/ignorefileglob2",
            "dockerignore/dir/ignorefile",
            "dockerignore/ignorefile",
            "dockerignore/ignorefilesuffix",
            "dockerignore/dir/file",
            "dockerignore/foo.sh",
            "dockerignore/unignore",
        ]
        for file in files:
            fs.touch(file)
        with fs.open("dockerignore/.dockerignore", "wt") as f:
            f.write(
                """
                # comment

                # dirs/files
                ignoredir
                ignorefile

                # globs
                */ignorefileglo*1
                **/ignorefileglob2
                dir?

                # inverse patterns
                unignore
                !unignore

                # ignore .
                .
            """
            )

        with _build_context("img", "memory://dockerignore") as f:
            with tarfile.open(fileobj=f, mode="r") as tf:
                self.assertCountEqual(
                    tf.getnames(),
                    {
                        "Dockerfile.torchx",
                        "foo.sh",
                        ".dockerignore",
                        "dir/ignorefile",
                        "ignorefilesuffix",
                        "dir/file",
                        "unignore",
                    },
                )
                self.assertGreater(tf.getmember("Dockerfile.torchx").size, 0)

    def test_dockerignore_include(self) -> None:
        fs = fsspec.filesystem("memory")
        files = [
            "dockerignore/Dockerfile.torchx",
            "dockerignore/timm_app.py",
            "dockerignore/bigfile",
            "dockerignore/some_dir/a",
            "dockerignore/some_dir/b",
            "dockerignore/some_dir/ignore1",
            "dockerignore/some_dir/ignore2",
            "dockerignore/some_dir/subdir/a",
            "dockerignore/some_dir/subdir/b",
            "dockerignore/some_dir/subdir/ignore3",
        ]
        for file in files:
            fs.touch(file)
        with fs.open("dockerignore/.dockerignore", "wt") as f:
            f.write(
                """
                    *
                    !timm_app.py
                    **/a
                    !some_dir
                    **/ignore1
                    some_dir/ignore2
                    some_dir/subdir/ignore3
                    """
            )

        with _build_context("img", "memory://dockerignore") as f:
            with tarfile.open(fileobj=f, mode="r") as tf:
                self.assertCountEqual(
                    tf.getnames(),
                    {
                        "Dockerfile.torchx",
                        "timm_app.py",
                        "some_dir/a",
                        "some_dir/b",
                        "some_dir/subdir/a",
                        "some_dir/subdir/b",
                    },
                )
                self.assertEqual(tf.getmember("Dockerfile.torchx").size, 0)
