#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os.path
import sys
import tempfile
import unittest
from typing import TypedDict, Optional
from unittest.mock import patch

import yaml
from torchx.container.main import main
from torchx.sdk.component import Component
from torchx.sdk.storage import temppath, upload_file, download_file


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


class NoopConfig(TypedDict):
    pass


class NoopInputs(TypedDict):
    pass


class NoopOutputs(TypedDict):
    pass


class NoopComponent(Component[NoopConfig, NoopInputs, NoopOutputs]):
    Version: str = "0.1"

    def run(self, inputs: NoopInputs, outputs: NoopOutputs) -> None:
        pass


class ContainerTest(unittest.TestCase):
    def test_main(self) -> None:
        main(
            [
                "main.py",
                "torchx.container.test.main_test.TestComponent",
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
                    "torchx.components.io.copy.Copy",
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

    def test_config_storage_providers(self) -> None:
        """
        Tests that storage providers from the specified config are loaded.
        """

        module = "torchx.container.test.dummy_module"
        config = {
            "storage_providers": [module],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "torchx.yaml")
            with open(config_path, "w") as f:
                yaml.dump(config, f)
            with patch("torchx.container.main.TORCHX_CONFIG_PATH", config_path):
                self.assertNotIn(module, sys.modules)
                main(
                    [
                        "main.py",
                        "torchx.container.test.main_test.NoopComponent",
                    ]
                )
                self.assertIn(module, sys.modules)
