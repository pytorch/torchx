# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings

try:
    from importlib import metadata
    from importlib.metadata import EntryPoint
except ImportError:
    import importlib_metadata as metadata
    from importlib_metadata import EntryPoint
from typing import Any, Dict, Optional


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

    entrypoints = metadata.entry_points()

    if group not in entrypoints and default:
        return default

    eps: Dict[str, EntryPoint] = {ep.name: ep for ep in entrypoints[group]}

    if name not in eps and default:
        return default
    else:
        ep = eps[name]
        return ep.load()


# pyre-ignore-all-errors[3, 2]
def load_group(
    group: str, default: Optional[Dict[str, Any]] = None, ignore_missing=False
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

    entrypoints = metadata.entry_points()

    if group not in entrypoints:
        return default

    eps = {}
    for ep in entrypoints[group]:
        try:
            eps[ep.name] = ep.load()
        except (ModuleNotFoundError, AttributeError) as e:
            if ignore_missing:
                warnings.warn(
                    f"{str(e)}, but ignore_missing={ignore_missing},"
                    f" skipping over `{ep.name} = {ep.value}`"
                )
            else:
                raise e
    return eps
