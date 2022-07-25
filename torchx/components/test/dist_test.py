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
        num_replicas_list = [
            1,
            1,
            2,
        ]  # number of replicas should be either nnodes or max_nnodes when it's elastic
        nnodes_rep_list = [
            "",
            "",
            "1:2",
        ]  # nnodes representation should only contain the nodes numbers or nothing if not elastic
        for i in range(3):
            app = dist.ddp(script="foo.py", j=j_list[i])
            self.assertEqual(app.roles[0].num_replicas, num_replicas_list[i])
            self.assertEqual(app.roles[0].nnodes_rep, nnodes_rep_list[i])
            
    def test_ddp_debug(self) -> None:
        app = dist.ddp(script="foo.py", debug=True)
        env = app.roles[0].env
        for k, v in dist._TORCH_DEBUG_FLAGS.items():
            self.assertEqual(env[k], v)
