#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torchx.specs as specs
from torchx.components.serve.serve import torchserve


class ServeTest(unittest.TestCase):
    def test_torchserve(self) -> None:
        want = specs.Application(
            name="torchx-serve-torchserve",
            roles=[
                specs.Role(
                    name="torchx-serve-torchserve",
                    entrypoint="python3",
                    args=[
                        "torchx/apps/serve/serve.py",
                        "--model_path",
                        "the_model_path",
                        "--management_api",
                        "http://localhost:1234",
                        "--initial_workers",
                        "1",
                    ],
                    container=specs.Container(
                        image="torchx:latest",
                        port_map={"model-download": 8222},
                    ),
                ),
            ],
        )
        out = torchserve(
            "the_model_path",
            "http://localhost:1234",
            image="torchx:latest",
            params={
                "initial_workers": "1",
            },
        )
        self.assertEqual(out, want)
