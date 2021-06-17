#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import TypedDict

from torchx.runtime.component import Component
from torchx.runtime.storage import upload_blob, download_blob


class Config(TypedDict):
    pass


class Inputs(TypedDict):
    input_path: str


class Outputs(TypedDict):
    output_path: str


class Copy(Component[Config, Inputs, Outputs]):
    Version: str = "0.1"

    def run(self, inputs: Inputs, outputs: Outputs) -> None:
        upload_blob(outputs["output_path"], download_blob(inputs["input_path"]))
