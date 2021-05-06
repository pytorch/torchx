#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchx.sdk.storage import upload_file, download_file, temppath


class StorageTest(unittest.TestCase):
    def test_file_provider(self) -> None:
        data = bytes(range(256))

        with temppath() as path:
            upload_file(path, data)
            out_data = download_file(path)

        self.assertEqual(out_data, data)
