# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

from torchx.specs.named_resources_aws import (
    aws_g4dn_12xlarge,
    aws_g4dn_16xlarge,
    aws_g4dn_2xlarge,
    aws_g4dn_4xlarge,
    aws_g4dn_8xlarge,
    aws_g4dn_metal,
    aws_g4dn_xlarge,
    aws_g5_12xlarge,
    aws_g5_16xlarge,
    aws_g5_24xlarge,
    aws_g5_2xlarge,
    aws_g5_48xlarge,
    aws_g5_4xlarge,
    aws_g5_8xlarge,
    aws_g5_xlarge,
    aws_m5_2xlarge,
    aws_p3_16xlarge,
    aws_p3_2xlarge,
    aws_p3_8xlarge,
    aws_p3dn_24xlarge,
    aws_p4d_24xlarge,
    aws_p4de_24xlarge,
    aws_p5_48xlarge,
    aws_t3_medium,
    aws_trn1_2xlarge,
    aws_trn1_32xlarge,
    EFA_DEVICE,
    GiB,
    K8S_ITYPE,
    NAMED_RESOURCES,
)


class NamedResourcesTest(unittest.TestCase):
    def test_aws_p3(self) -> None:
        p3_2 = aws_p3_2xlarge()
        self.assertEqual(8, p3_2.cpu)
        self.assertEqual(1, p3_2.gpu)
        self.assertEqual(61 * GiB, p3_2.memMB)

        p3_8 = aws_p3_8xlarge()
        self.assertEqual(p3_8.cpu, p3_2.cpu * 4)
        self.assertEqual(p3_8.gpu, p3_2.gpu * 4)
        self.assertEqual(p3_8.memMB, p3_2.memMB * 4)

        p3_16 = aws_p3_16xlarge()
        self.assertEqual(p3_16.cpu, p3_2.cpu * 8)
        self.assertEqual(p3_16.gpu, p3_2.gpu * 8)
        self.assertEqual(p3_16.memMB, p3_2.memMB * 8)

        p3dn_24 = aws_p3dn_24xlarge()
        self.assertEqual(96, p3dn_24.cpu)
        self.assertEqual(p3_16.gpu, p3dn_24.gpu)
        self.assertEqual(768 * GiB, p3dn_24.memMB)
        self.assertEqual({EFA_DEVICE: 1}, p3dn_24.devices)

    def test_aws_p4(self) -> None:
        p4d = aws_p4d_24xlarge()
        p4de = aws_p4de_24xlarge()

        self.assertEqual(96, p4d.cpu)
        self.assertEqual(8, p4d.gpu)
        self.assertEqual(1152 * GiB, p4d.memMB)
        self.assertEqual({EFA_DEVICE: 4}, p4d.devices)

        self.assertEqual(p4de.cpu, p4d.cpu)
        self.assertEqual(p4de.gpu, p4d.gpu)
        self.assertEqual(p4de.memMB, p4d.memMB)
        self.assertEqual({EFA_DEVICE: 4}, p4de.devices)

    def test_aws_p5(self) -> None:
        p5 = aws_p5_48xlarge()

        self.assertEqual(192, p5.cpu)
        self.assertEqual(8, p5.gpu)
        self.assertEqual(2048 * GiB, p5.memMB)
        self.assertEqual({EFA_DEVICE: 32}, p5.devices)

    def test_aws_g4dn(self) -> None:
        g4d = aws_g4dn_xlarge()
        self.assertEqual(4, g4d.cpu)
        self.assertEqual(1, g4d.gpu)
        self.assertEqual(16 * GiB, g4d.memMB)

        g4d_2 = aws_g4dn_2xlarge()
        self.assertEqual(g4d_2.cpu, g4d.cpu * 2)
        self.assertEqual(g4d_2.gpu, g4d.gpu)
        self.assertEqual(g4d_2.memMB, g4d.memMB * 2)

        g4d_4 = aws_g4dn_4xlarge()
        self.assertEqual(g4d_4.cpu, g4d.cpu * 4)
        self.assertEqual(g4d_4.gpu, g4d.gpu)
        self.assertEqual(g4d_4.memMB, g4d.memMB * 4)

        g4d_8 = aws_g4dn_8xlarge()
        self.assertEqual(g4d_8.cpu, g4d.cpu * 8)
        self.assertEqual(g4d_8.gpu, g4d.gpu)
        self.assertEqual(g4d_8.memMB, g4d.memMB * 8)

        g4d_16 = aws_g4dn_16xlarge()
        self.assertEqual(g4d_16.cpu, g4d.cpu * 16)
        self.assertEqual(g4d_16.gpu, g4d.gpu)
        self.assertEqual(g4d_16.memMB, g4d.memMB * 16)

        g4d_12 = aws_g4dn_12xlarge()
        self.assertEqual(g4d_12.cpu, g4d.cpu * 12)
        self.assertEqual(g4d_12.gpu, 4)
        self.assertEqual(g4d_12.memMB, g4d.memMB * 12)

        g4d_metal = aws_g4dn_metal()
        self.assertEqual(g4d_metal.cpu, g4d_12.cpu * 2)
        self.assertEqual(g4d_metal.gpu, g4d_12.gpu * 2)
        self.assertEqual(g4d_metal.memMB, g4d_12.memMB * 2)

    def test_aws_g5(self) -> None:
        g5 = aws_g5_xlarge()

        self.assertEqual(4, g5.cpu)
        self.assertEqual(1, g5.gpu)
        self.assertEqual(16 * GiB, g5.memMB)

        g5_2 = aws_g5_2xlarge()
        self.assertEqual(g5_2.cpu, g5.cpu * 2)
        self.assertEqual(g5_2.gpu, g5.gpu)
        self.assertEqual(g5_2.memMB, g5.memMB * 2)

        g5_4 = aws_g5_4xlarge()
        self.assertEqual(g5_4.cpu, g5.cpu * 4)
        self.assertEqual(g5_4.gpu, g5.gpu)
        self.assertEqual(g5_4.memMB, g5.memMB * 4)

        g5_8 = aws_g5_8xlarge()
        self.assertEqual(g5_8.cpu, g5.cpu * 8)
        self.assertEqual(g5_8.gpu, g5.gpu)
        self.assertEqual(g5_8.memMB, g5.memMB * 8)

        g5_16 = aws_g5_16xlarge()
        self.assertEqual(g5_16.cpu, g5.cpu * 16)
        self.assertEqual(g5_16.gpu, g5.gpu)
        self.assertEqual(g5_16.memMB, g5.memMB * 16)

        g5_12 = aws_g5_12xlarge()
        self.assertEqual(48, g5_12.cpu)
        self.assertEqual(4, g5_12.gpu)
        self.assertEqual(384 * GiB, g5_12.memMB * 2)

        g5_24 = aws_g5_24xlarge()
        self.assertEqual(g5_24.cpu, g5_12.cpu * 2)
        self.assertEqual(g5_24.gpu, g5_12.gpu)
        self.assertEqual(g5_24.memMB, g5_12.memMB * 2)

        g5_48 = aws_g5_48xlarge()
        self.assertEqual(g5_48.cpu, g5_12.cpu * 4)
        self.assertEqual(g5_48.gpu, 8)
        self.assertEqual(g5_48.memMB, g5_12.memMB * 4)

    def test_aws_trn1(self) -> None:
        trn1_2 = aws_trn1_2xlarge()

        self.assertEqual(8, trn1_2.cpu)
        self.assertEqual(0, trn1_2.gpu)
        self.assertEqual(32 * GiB, trn1_2.memMB)

        trn1_32 = aws_trn1_32xlarge()
        self.assertEqual(trn1_32.cpu, trn1_2.cpu * 16)
        self.assertEqual(trn1_32.gpu, trn1_2.gpu)
        self.assertEqual(trn1_32.memMB, trn1_2.memMB * 16)

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

    def test_capabilities(self) -> None:
        for name, func in NAMED_RESOURCES.items():
            resource = func()
            self.assertIn(K8S_ITYPE, resource.capabilities)
