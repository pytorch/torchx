# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from unittest import mock
from unittest.mock import MagicMock

from torchx.util.cuda import has_cuda_devices

TORCH_CUDA_IS_AVAIL = "torch.cuda.is_available"
TORCH_CUDA_DEVICE_COUNT = "torch.cuda.device_count"


class CudaTest(unittest.TestCase):
    @mock.patch(TORCH_CUDA_IS_AVAIL, return_value=True)
    def test_has_cuda_devices_cuda_is_available(self, _: MagicMock) -> None:
        with mock.patch(TORCH_CUDA_DEVICE_COUNT, return_value=1):
            self.assertTrue(has_cuda_devices())

        with mock.patch(TORCH_CUDA_DEVICE_COUNT, return_value=0):
            self.assertFalse(has_cuda_devices())

    @mock.patch(TORCH_CUDA_IS_AVAIL, return_value=False)
    def test_has_cuda_devices_cuda_is_not_available(self, _: MagicMock) -> None:
        with mock.patch(TORCH_CUDA_DEVICE_COUNT, return_value=1):
            self.assertFalse(has_cuda_devices())

        with mock.patch(TORCH_CUDA_DEVICE_COUNT, return_value=0):
            self.assertFalse(has_cuda_devices())
