# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

from torchx.specs.named_resources_aws import (
    aws_p3_2xlarge,
    aws_p3_8xlarge,
    aws_m5_2xlarge,
    aws_t3_medium,
    GiB,
)


class NamedResourcesTest(unittest.TestCase):
    def test_aws_p3_2xlarge(self) -> None:
        resource = aws_p3_2xlarge()
        self.assertEqual(8, resource.cpu)
        self.assertEqual(1, resource.gpu)
        self.assertEqual(61 * GiB, resource.memMB)

    def test_aws_p3_8xlarge(self) -> None:
        resource = aws_p3_8xlarge()
        self.assertEqual(32, resource.cpu)
        self.assertEqual(4, resource.gpu)
        self.assertEqual(244 * GiB, resource.memMB)

    def test_aws_m5_2xlarge(self) -> None:
        resource = aws_m5_2xlarge()
        self.assertEqual(8, resource.cpu)
        self.assertEqual(0, resource.gpu)
        self.assertEqual(32 * GiB, resource.memMB)

    def test_aws_t3_medium(self) -> None:
        resource = aws_t3_medium()
        self.assertEqual(2, resource.cpu)
        self.assertEqual(0, resource.gpu)
        self.assertEqual(4 * GiB, resource.memMB)
