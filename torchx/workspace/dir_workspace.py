#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import posixpath
import shutil
from tempfile import mkdtemp
from typing import Mapping

import fsspec
from torchx.specs import CfgVal, Role
from torchx.workspace.api import walk_workspace, Workspace


class TmpDirWorkspace(Workspace):
    def build_workspace_and_update_role(
        self, role: Role, workspace: str, cfg: Mapping[str, CfgVal]
    ) -> None:
        """
        Creates a new temp directory from the workspace. Role image fields will
        be set to the ``job_dir``.

        Any files listed in the ``.torchxignore`` folder will be skipped.
        """
        job_dir = mkdtemp(prefix="torchx_workspace")
        _copy_to_dir(workspace, job_dir)
        role.image = job_dir


class DirWorkspace(Workspace):
    def build_workspace_and_update_role(
        self, role: Role, workspace: str, cfg: Mapping[str, CfgVal]
    ) -> None:
        """
        Creates a new directory specified by ``job_dir`` from the workspace. Role
        image fields will be set to the ``job_dir``.

        Any files listed in the ``.torchxignore`` folder will be skipped.
        """
        job_dir = cfg.get("job_dir")
        if job_dir is None:
            return
        assert isinstance(job_dir, str), "job_dir must be str"

        os.mkdir(job_dir)
        _copy_to_dir(workspace, job_dir)
        role.image = job_dir


def _copy_to_dir(workspace: str, target: str) -> None:
    fs, path = fsspec.core.url_to_fs(workspace)
    assert isinstance(path, str), "path must be str"

    for dir, dirs, files in walk_workspace(fs, path):
        assert isinstance(dir, str), "path must be str"
        relpath = posixpath.relpath(dir, path)
        for file, info in files.items():
            filepath = posixpath.join(
                target,
                posixpath.join(relpath, file) if relpath != "." else file,
            )
            with fs.open(info["name"], "rb") as src, fsspec.open(filepath, "wb") as dst:
                shutil.copyfileobj(src, dst)
