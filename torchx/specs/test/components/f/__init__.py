#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import functools

from torchx import specs

from .g import Cls


@functools.wraps(Cls)
def comp_f(**kwargs) -> specs.AppDef:  # pyre-ignore[2]
    return Cls(**kwargs).build()
