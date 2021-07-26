# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch

from torchx.components.base.roles import create_torch_dist_role
from torchx.components.base.torch_dist_component import torch_dist_component
from torchx.specs import api


class TorchDistComponentTest(unittest.TestCase):
    def test_torch_dist_component(self) -> None:
        want = api.AppDef(
            name="datapreproc",
            roles=[
                api.Role(
                    name="worker",
                    image="custom:latest",
                    base_image="pytorch/pytorch:latest",
                    entrypoint="python",
                    resource=api.NULL_RESOURCE,
                    args=[
                        "-m",
                        "torch.distributed.launch",
                        "--nnodes",
                        "1",
                        "--nproc_per_node",
                        "1",
                        "--rdzv_backend",
                        "etcd",
                        "--rdzv_id",
                        "${app_id}",
                        "--role",
                        "worker",
                        "${img_root}/python3",
                        "--version",
                    ],
                    env={"FOO": "bar"},
                    num_replicas=1,
                ),
            ],
        )
        args = ["--version"]
        with patch("torchx.components.base.load") as create_role_mock:
            create_role_mock.return_value = create_torch_dist_role
            out = torch_dist_component(
                name="datapreproc",
                image="custom:latest",
                base_image="pytorch/pytorch:latest",
                entrypoint="python3",
                env={
                    "FOO": "bar",
                },
                args=args,
                nnodes=1,
                nproc_per_node=1,
            )
            self.assertEqual(out, want)
