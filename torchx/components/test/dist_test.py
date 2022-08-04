# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchx.components.dist as dist
from torchx.components.component_test_base import ComponentTestCase


class DistributedComponentTest(ComponentTestCase):
    def test_ddp(self) -> None:
        self.validate(dist, "ddp")

    def test_ddp_mounts(self) -> None:
        app = dist.ddp(
            script="foo.py", mounts=["type=bind", "src=/dst", "dst=/dst", "readonly"]
        )
        self.assertEqual(len(app.roles[0].mounts), 1)

    def test_ddp_parse_j(self) -> None:
        """test samples for different forms of -j {nnodes}x{nproc_per_node}"""
        j_list = ["1", "1x2", "1:2x3"]
        min_nnodes_list = [
            "1",
            "1",
            "1",
        ]  # minimum nnodes
        max_nnodes_list = [
            "1",
            "1",
            "2",
        ]  # maximum nnodes
        nnodes_rep_list = [
            "1",
            "1",
            "1:2",
        ]  # nnodes representation
        nproc_per_node_list = ["1", "2", "3"]
        for i in range(3):
            min_nnodes, max_nnodes, nproc_per_node, nnodes_rep = dist.parse_nnodes(
                j_list[i]
            )
            self.assertEqual(min_nnodes, min_nnodes_list[i])
            self.assertEqual(max_nnodes, max_nnodes_list[i])
            self.assertEqual(nproc_per_node, nproc_per_node_list[i])
            self.assertEqual(nnodes_rep, nnodes_rep_list[i])

    def test_ddp_parse_j_exception(self) -> None:
        j_exception = ["1x", "x2", ":3", ":2x1", "1x2:3"]
        for j in j_exception:
            with self.assertRaises(ValueError):
                dist.parse_nnodes(j)

    def test_ddp_debug(self) -> None:
        app = dist.ddp(script="foo.py", debug=True)
        env = app.roles[0].env
        for k, v in dist._TORCH_DEBUG_FLAGS.items():
            self.assertEqual(env[k], v)
