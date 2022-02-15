# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import posixpath
import tarfile
import tempfile
from typing import IO

import fsspec
import torchx
from torchx.specs import Role
from torchx.workspace.api import Workspace


class DockerWorkspace(Workspace):
    LABEL_VERSION: str = "torchx.pytorch.org/version"

    def build_workspace_and_update_role(self, role: Role, workspace: str) -> None:
        """
        Builds a new docker image using the ``role``'s image as the base image
        and updates the ``role``'s image with this newly built docker image id

        Args:
            role: the role whose image (a Docker image) is to be used as the base image
            workspace: a fsspec path to a directory with contents to be overlaid
        """

        context = _build_context(role.image, workspace)

        try:
            import docker

            image, logs = docker.from_env().images.build(
                fileobj=context,
                custom_context=True,
                pull=True,
                rm=True,
                labels={
                    self.LABEL_VERSION: torchx.__version__,
                },
            )
            role.image = image.id
        finally:
            context.close()


def _build_context(img: str, workspace: str) -> IO[bytes]:
    # f is closed by parent, NamedTemporaryFile auto closes on GC
    f = tempfile.NamedTemporaryFile(  # noqa P201
        prefix="torchx-context",
        suffix=".tar",
    )
    dockerfile = bytes(f"FROM {img}\nCOPY . .\n", encoding="utf-8")
    with tarfile.open(fileobj=f, mode="w") as tf:
        info = tarfile.TarInfo("Dockerfile")
        info.size = len(dockerfile)
        tf.addfile(info, io.BytesIO(dockerfile))

        _copy_to_tarfile(workspace, tf)

    f.seek(0)
    return f


def _copy_to_tarfile(workspace: str, tf: tarfile.TarFile) -> None:
    # TODO(d4l3k) implement docker ignore files

    fs, path = fsspec.core.url_to_fs(workspace)
    assert isinstance(path, str), "path must be str"

    for dir, dirs, files in fs.walk(path, detail=True):
        assert isinstance(dir, str), "path must be str"
        relpath = posixpath.relpath(dir, path)
        for file, info in files.items():
            with fs.open(info["name"], "rb") as f:
                tinfo = tarfile.TarInfo(posixpath.join(relpath, file))
                tinfo.size = info["size"]
                tf.addfile(tinfo, f)
