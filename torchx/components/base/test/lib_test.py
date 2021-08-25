#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata

from unittest.mock import patch

from torchx.components.base import torch_dist_role
from torchx.specs.api import Resource


class TorchxBaseLibTest(unittest.TestCase):
    def test_torch_dist_role_str_resources(self) -> None:
        expected_resources = {}
        with patch.object(metadata, "entry_points") as entry_points_mock:
            entry_points_mock.return_value = {}
            with self.assertRaises(KeyError):
                torch_dist_role(
                    name="dist_role",
                    image="test_image",
                    entrypoint="test_entry.py",
                    base_image="test_base_image",
                    resource="unknown resource",
                )

    def test_torch_dist_role_default(self) -> None:
        with patch.object(metadata, "entry_points") as entry_points_mock:
            entry_points_mock.return_value = {}
            role = torch_dist_role(
                name="dist_role",
                image="test_image",
                entrypoint="test_entry.py",
                base_image="test_base_image",
                resource=Resource(1, 1, 10),
                args=["arg1", "arg2"],
                env={"FOO": "BAR"},
                nnodes=2,
            )

        self.assertEqual("python", role.entrypoint)
        expected_args = [
            "-m",
            "torch.distributed.run",
            "--nnodes",
            "2",
            "--rdzv_backend",
            "etcd",
            "--rdzv_id",
            "${app_id}",
            "--role",
            "dist_role",
            "test_entry.py",
            "arg1",
            "arg2",
        ]
        self.assertListEqual(expected_args, role.args)
