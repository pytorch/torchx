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
        self.assertEqual(dist.parse_nnodes("2"), (1, 1, 2, "1"))  # nproc_per_node is 2
        self.assertEqual(dist.parse_nnodes("1x2"), (1, 1, 2, "1"))
        self.assertEqual(dist.parse_nnodes("1:2x3"), (1, 2, 3, "1:2"))

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
