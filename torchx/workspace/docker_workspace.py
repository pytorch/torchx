# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import io
import logging
import posixpath
import tarfile
import tempfile
from typing import IO, TYPE_CHECKING, Optional, Dict, Tuple, Mapping

import fsspec
import torchx
from torchx.specs import Role, AppDef, CfgVal
from torchx.workspace.api import Workspace

if TYPE_CHECKING:
    from docker import DockerClient

log: logging.Logger = logging.getLogger(__name__)


class DockerWorkspace(Workspace):
    """
    DockerWorkspace will build patched docker images from the workspace.
    """

    LABEL_VERSION: str = "torchx.pytorch.org/version"

    def __init__(self, docker_client: Optional["DockerClient"] = None) -> None:
        self.__docker_client = docker_client

    @property
    def _docker_client(self) -> "DockerClient":
        client = self.__docker_client
        if client is None:
            import docker

            client = docker.from_env()
            self.__docker_client = client
        return client

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

            image, _ = self._docker_client.images.build(
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

    def _update_app_images(
        self, app: AppDef, cfg: Mapping[str, CfgVal]
    ) -> Dict[str, Tuple[str, str]]:
        HASH_PREFIX = "sha256:"

        images_to_push = {}
        for role in app.roles:
            if role.image.startswith(HASH_PREFIX):
                image_repo = cfg.get("image_repo")
                if not image_repo:
                    raise KeyError(
                        f"must specify the image repository via `image_repo` config to be able to upload local image {role.image}"
                    )
                assert isinstance(image_repo, str), "image_repo must be str"

                image_hash = role.image[len(HASH_PREFIX) :]
                remote_image = image_repo + ":" + image_hash
                images_to_push[role.image] = (
                    image_repo,
                    image_hash,
                )
                role.image = remote_image
        return images_to_push

    def _push_images(self, images_to_push: Dict[str, Tuple[str, str]]) -> None:
        if len(images_to_push) == 0:
            return

        client = self._docker_client
        for local, (repo, tag) in images_to_push.items():
            log.info(f"pushing image {repo}:{tag}...")
            img = client.images.get(local)
            img.tag(repo, tag=tag)
            for line in client.images.push(repo, tag=tag, stream=True, decode=True):
                ERROR_KEY = "error"
                if ERROR_KEY in line:
                    raise RuntimeError(
                        f"failed to push docker image: {line[ERROR_KEY]}"
                    )
                log.info(f"docker: {line}")


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
