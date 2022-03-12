# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import fnmatch
import posixpath
from typing import TYPE_CHECKING, Iterable, Mapping, Tuple

from torchx.specs import Role, CfgVal

if TYPE_CHECKING:
    from fsspec import AbstractFileSystem

TORCHX_IGNORE = ".torchxignore"


class Workspace(abc.ABC):
    """
    Note: (Prototype) this interface may change without notice!

    A mix-in that can be attached to a Scheduler that adds the ability to
    builds a workspace. A workspace is the local checkout of the codebase/project
    that builds into an image. The workspace scheduler adds capability to
    automatically rebuild images or generate diff patches that are
    applied to the ``Role``, allowing the user to make local code
    changes to the application and having those changes be reflected
    (either through a new image or an overlaid patch) at runtime
    without a manual image rebuild. The exact semantics of what the
    workspace build artifact is, is implementation dependent.
    """

    @abc.abstractmethod
    def build_workspace_and_update_role(
        self, role: Role, workspace: str, cfg: Mapping[str, CfgVal]
    ) -> None:
        """
        Builds the specified ``workspace`` with respect to ``img``
        and updates the ``role`` to reflect the built workspace artifacts.
        In the simplest case, this method builds a new image and updates
        the role's image. Certain (more efficient) implementations build
        incremental diff patches that overlay on top of the role's image.

        Note: this method mutates the passed ``role``.
        """
        ...


def _ignore(s: str, patterns: Iterable[str]) -> bool:
    match = False
    for pattern in patterns:
        if pattern.startswith("!") and fnmatch.fnmatch(s, pattern[1:]):
            match = False
        elif fnmatch.fnmatch(s, pattern):
            match = True
    return match


def walk_workspace(
    fs: "AbstractFileSystem",
    path: str,
    ignore_name: str = TORCHX_IGNORE,
) -> Iterable[Tuple[str, Iterable[str], Mapping[str, Mapping[str, object]]]]:
    """
    walk_workspace walks the filesystem path and applies the ignore rules
    specified via ``ignore_name``.
    This follows the rules for ``.dockerignore``.
    https://docs.docker.com/engine/reference/builder/#dockerignore-file
    """
    ignore_patterns = []
    ignore_path = posixpath.join(path, ignore_name)
    if fs.exists(ignore_path):
        with fs.open(ignore_path, "rt") as f:
            lines = f.readlines()
        for line in lines:
            line, _, _ = line.partition("#")
            line = line.strip()
            if len(line) == 0 or line == ".":
                continue
            ignore_patterns.append(line)

    for dir, dirs, files in fs.walk(path, detail=True):
        assert isinstance(dir, str), "path must be str"
        relpath = posixpath.relpath(dir, path)
        if _ignore(relpath, ignore_patterns):
            continue
        dirs = [
            d for d in dirs if not _ignore(posixpath.join(relpath, d), ignore_patterns)
        ]
        files = {
            file: info
            for file, info in files.items()
            if not _ignore(
                posixpath.join(relpath, file) if relpath != "." else file,
                ignore_patterns,
            )
        }
        yield dir, dirs, files
