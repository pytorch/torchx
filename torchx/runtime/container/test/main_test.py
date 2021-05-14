#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os.path
import tempfile
import unittest
from typing import TypedDict, Optional

from torchx.runtime.component import Component
from torchx.runtime.container.main import main
from torchx.runtime.storage import temppath, upload_file, download_file


class SubConfig(TypedDict):
    foo: str


class Config(TypedDict):
    integer: int
    float: float
    dict: SubConfig
    opt_dict: Optional[SubConfig]
    opt: Optional[int]
    opt_empty: Optional[int]


class Inputs(TypedDict):
    input_path: str


class Outputs(TypedDict):
    output_path: str


class TestComponent(Component[Config, Inputs, Outputs]):
    Version: str = "0.1"

    ran: bool = False

    def run(self, inputs: Inputs, outputs: Outputs) -> None:
        assert self.config["integer"] == 1
        assert self.config["float"] == 1.5
        assert self.config["dict"]["foo"] == "bar"
        assert self.config["opt"] == 2
        assert self.config.get("opt_empty") is None

        assert inputs["input_path"] == "somepath"
        assert outputs["output_path"] == "somepath2"

        TestComponent.ran = True


class ContainerTest(unittest.TestCase):
    def test_main(self) -> None:
        main(
            [
                "main.par",
                "torchx.runtime.container.test.main_test.TestComponent",
                "--input_path",
                "somepath",
                "--output_path",
                "somepath2",
                "--dict",
                json.dumps({"foo": "bar"}),
                "--integer",
                "1",
                "--float",
                "1.5",
                "--opt",
                "2",
            ]
        )
        self.assertTrue(TestComponent.ran)

    def test_output_path(self) -> None:
        with temppath() as input_path, temppath() as output_path, tempfile.TemporaryDirectory() as tmpdir:
            out_path_file = os.path.join(tmpdir, "dir", "out_path.txt")
            data = b"banana"
            upload_file(input_path, data)
            main(
                [
                    "main.par",
                    "torchx.apps.io.copy.Copy",
                    "--input_path",
                    input_path,
                    "--output_path",
                    output_path,
                    "--output-path-output_path",
                    out_path_file,
                ]
            )
            out = download_file(output_path)
            self.assertEqual(out, data)
            with open(out_path_file, "rt") as f:
                self.assertEqual(f.read(), output_path)
