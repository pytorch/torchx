# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, Optional

import importlib_metadata as metadata
from importlib_metadata import EntryPoint


# pyre-ignore-all-errors[3, 2]
def load(group: str, name: str, default=None):
    """
    Loads the entry point specified by

    ::

     [group]
     name1 = this.is:a_function
     -- or --
     name2 = this.is.a.module

    In case such an entry point is not found, an optional
    default is returned. If the default is not specified
    and the entry point is not found, then this method
    raises an error.
    """

    entrypoints = metadata.entry_points().select(group=group)

    if name not in entrypoints.names and default is not None:
        return default

    ep = entrypoints[name]
    return ep.load()


def _defer_load_ep(ep: EntryPoint) -> object:
    def run(*args: object, **kwargs: object) -> object:
        return ep.load()(*args, **kwargs)

    return run


# pyre-ignore-all-errors[3, 2]
def load_group(
    group: str,
    default: Optional[Dict[str, Any]] = None,
):
    """
    Loads all the entry points specified by ``group`` and returns
    the entry points as a map of ``name (str) -> entrypoint.load()``.

    For the following ``entry_point.txt``:

    ::

     [foo]
     bar = this.is:a_fn
     baz = this.is:b_fn

    1. ``load_group("foo")`` -> ``{"bar", this.is.a_fn, "baz": this.is.b_fn}``
    1. ``load_group("food")`` -> ``None``
    1. ``load_group("food", default={"hello": this.is.c_fn})`` -> ``{"hello": this.is.c_fn}``

    """

    entrypoints = metadata.entry_points().select(group=group)

    if len(entrypoints) == 0:
        return default

    eps = {}
    for ep in entrypoints:
        eps[ep.name] = _defer_load_ep(ep)
    return eps
