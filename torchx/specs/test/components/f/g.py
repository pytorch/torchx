#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict
from dataclasses import dataclass

import torchx
from torchx import specs

from .h import fake_decorator


@dataclass
class Args:
    name: str


@dataclass
class Cls(Args):
    def build(self) -> specs.AppDef:
        return specs.AppDef(
            name=self.name,
            roles=[
                specs.Role(
                    name=self.name,
                    image=torchx.IMAGE,
                    entrypoint="echo",
                    args=["hello world"],
                )
            ],
        )


@fake_decorator
def comp_g() -> specs.AppDef:
    return specs.AppDef(
        name="g",
        roles=[
            specs.Role(
                name="g",
                image=torchx.IMAGE,
                entrypoint="echo",
                args=["hello world"],
            )
        ],
    )
