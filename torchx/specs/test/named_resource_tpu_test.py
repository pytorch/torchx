# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

from torchx.specs import Resource
from torchx.specs import named_resources_tpu as tpu


class NamedResourcesTest(unittest.TestCase):
    def test_tf_version(self) -> None:
        self.assertEqual(tpu._get_tf_version("2.123.0+cu102"), "pytorch-2.123")
        self.assertEqual(
            tpu._get_tf_version("1.12.0.dev20220419+cu113"), "pytorch-nightly"
        )

    def test_tpu_v3_8(self) -> None:
        want = Resource(
            cpu=0,
            memMB=0,
            gpu=0,
            capabilities={
                "tf-version.cloud-tpus.google.com": "pytorch-1.11",
            },
            devices={
                "cloud-tpus.google.com/v3": 8,
            },
        )
        self.assertEqual(tpu.tpu_v3_8(), want)
        self.assertEqual(tpu.NAMED_RESOURCES["tpu_v3_8"](), want)

    def test_tpu_v3_2048(self) -> None:
        want = Resource(
            cpu=0,
            memMB=0,
            gpu=0,
            capabilities={
                "tf-version.cloud-tpus.google.com": "pytorch-1.11",
            },
            devices={
                "cloud-tpus.google.com/v3": 2048,
            },
        )
        self.assertEqual(tpu.tpu_v3_2048(), want)
        self.assertEqual(tpu.NAMED_RESOURCES["tpu_v3_2048"](), want)

    def test_tpu_v2_8(self) -> None:
        want = Resource(
            cpu=0,
            memMB=0,
            gpu=0,
            capabilities={
                "tf-version.cloud-tpus.google.com": "pytorch-1.11",
            },
            devices={
                "cloud-tpus.google.com/v2": 8,
            },
        )
        self.assertEqual(tpu.tpu_v2_8(), want)
        self.assertEqual(tpu.NAMED_RESOURCES["tpu_v2_8"](), want)

    def test_tpu_preemptible_v2_8(self) -> None:
        want = Resource(
            cpu=0,
            memMB=0,
            gpu=0,
            capabilities={
                "tf-version.cloud-tpus.google.com": "pytorch-1.11",
            },
            devices={
                "cloud-tpus.google.com/preemptible-v2": 8,
            },
        )
        self.assertEqual(tpu.tpu_preemptible_v2_8(), want)
        self.assertEqual(tpu.NAMED_RESOURCES["tpu_preemptible_v2_8"](), want)

    def test_tpu_preemptible_v3_8(self) -> None:
        want = Resource(
            cpu=0,
            memMB=0,
            gpu=0,
            capabilities={
                "tf-version.cloud-tpus.google.com": "pytorch-1.11",
            },
            devices={
                "cloud-tpus.google.com/preemptible-v3": 8,
            },
        )
        self.assertEqual(tpu.tpu_preemptible_v3_8(), want)
        self.assertEqual(tpu.NAMED_RESOURCES["tpu_preemptible_v3_8"](), want)
