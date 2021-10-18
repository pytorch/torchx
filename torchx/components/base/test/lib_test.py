#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

from torchx.components.base import torch_dist_role
from torchx.specs.api import Resource


try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata


class TorchxBaseLibTest(unittest.TestCase):
    def test_torch_dist_role_str_resources(self) -> None:
        with patch.object(metadata, "entry_points") as entry_points_mock:
            entry_points_mock.return_value = {}
            with self.assertRaises(KeyError):
                torch_dist_role(
                    name="dist_role",
                    image="test_image",
                    entrypoint="test_entry.py",
                    resource="unknown resource",
                )

    def test_torch_dist_role(self) -> None:
        with patch.object(metadata, "entry_points") as entry_points_mock:
            entry_points_mock.return_value = {}
            role = torch_dist_role(
                name="dist_role",
                image="test_image",
                entrypoint="test_entry.py",
                resource=Resource(1, 1, 10),
                args=["arg1", "arg2"],
                env={"FOO": "BAR"},
                num_replicas=2,
            )

        self.assertEqual("python", role.entrypoint)
        expected_args = [
            "-m",
            "torch.distributed.run",
            "--rdzv_backend",
            "c10d",
            "--rdzv_endpoint",
            "localhost:29500",
            "--rdzv_id",
            "${app_id}",
            "--role",
            "dist_role",
            "--nnodes",
            "2",
            "test_entry.py",
            "arg1",
            "arg2",
        ]
        self.assertListEqual(expected_args, role.args)

    def test_torch_dist_role_nnodes_override(self) -> None:
        with patch.object(metadata, "entry_points") as entry_points_mock:
            entry_points_mock.return_value = {}
            role = torch_dist_role(
                name="dist_role",
                image="test_image",
                entrypoint="test_entry.py",
                resource=Resource(1, 1, 10),
                args=["arg1", "arg2"],
                env={"FOO": "BAR"},
                num_replicas=2,
                nnodes=10,
            )

        self.assertEqual("python", role.entrypoint)
        expected_args = [
            "-m",
            "torch.distributed.run",
            "--nnodes",
            "10",
            "--rdzv_backend",
            "c10d",
            "--rdzv_endpoint",
            "localhost:29500",
            "--rdzv_id",
            "${app_id}",
            "--role",
            "dist_role",
            "test_entry.py",
            "arg1",
            "arg2",
        ]
        self.assertListEqual(expected_args, role.args)
