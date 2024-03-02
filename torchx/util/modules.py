# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import importlib
from types import ModuleType
from typing import Callable, Optional, Union


def load_module(path: str) -> Union[ModuleType, Optional[Callable[..., object]]]:
    """
    Loads and returns the module/module attr represented by the ``path``: ``full.module.path:optional_attr``

    ::


    1. ``load_module("this.is.a_module:fn")`` -> equivalent to ``this.is.a_module.fn``
    1. ``load_module("this.is.a_module")`` -> equivalent to ``this.is.a_module``
    """
    parts = path.split(":", 2)
    module_path, method = parts[0], parts[1] if len(parts) > 1 else None
    module = None
    i, n = -1, len(module_path)
    try:
        while i < n:
            i = module_path.find(".", i + 1)
            i = i if i >= 0 else n
            module = importlib.import_module(module_path[:i])
        return getattr(module, method) if method else module
    except Exception:
        return None
