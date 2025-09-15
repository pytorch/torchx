#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict


import functools
from typing import Any, Callable


def fake_decorator(  # pyre-ignore[3]
    func: Callable[..., Any],  # pyre-ignore[2]
) -> Callable[..., Any]:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Callable[..., Any]:  # pyre-ignore[3]
        # Fake decorator: just calls the original function
        return func(*args, **kwargs)

    return wrapper
