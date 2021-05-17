#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os.path
import tempfile
import unittest

from torchx.runtime.storage import (
    upload_blob,
    download_blob,
    upload_file,
    download_file,
    temppath,
)


class StorageTest(unittest.TestCase):
    def test_file_provider_blob(self) -> None:
        data = bytes(range(256))

        with temppath() as path:
            upload_blob(path, data)
            out_data = download_blob(path)

        self.assertEqual(out_data, data)

    def test_file_provider_file(self) -> None:
        data = bytes(range(256))

        with temppath() as remote_path, tempfile.TemporaryDirectory() as tmpdir:
            upload_path = os.path.join(tmpdir, "upload")
            download_path = os.path.join(tmpdir, "download")
            with open(upload_path, "wb") as f:
                f.write(data)

            upload_file(upload_path, remote_path)
            download_file(remote_path, download_path)

            with open(download_path, "rb") as f:
                out_data = f.read()

        self.assertEqual(out_data, data)
