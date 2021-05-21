# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from importlib import metadata
from importlib.metadata import EntryPoint
from typing import Dict


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
