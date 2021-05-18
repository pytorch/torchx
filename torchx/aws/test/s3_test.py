# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os.path
import tempfile
import unittest
import unittest.mock as mock
from typing import Dict

import boto3
from moto import mock_s3
from torchx.aws.s3 import init_plugin
from torchx.runtime.storage import (
    _PROVIDERS,
    StorageProvider,
    download_blob,
    download_file,
    upload_blob,
    upload_file,
)


_EMPTY_MAP: Dict[str, StorageProvider] = {}


class S3Test(unittest.TestCase):
    def _create_bucket(self) -> None:
        region = "us-west-2"
        client = boto3.Session().client("s3", region_name=region)
        client.create_bucket(
            Bucket="bucket", CreateBucketConfiguration={"LocationConstraint": region}
        )

    @mock.patch.dict(_PROVIDERS, _EMPTY_MAP)
    @mock_s3
    def test_storage_provider_blob(self) -> None:
        self._create_bucket()
        init_plugin(None)
        path = "s3://bucket/path"
        body = b"foo"
        upload_blob(path, body)
        self.assertEqual(download_blob(path), body)

    @mock.patch.dict(_PROVIDERS, _EMPTY_MAP)
    @mock_s3
    def test_storage_provider_file(self) -> None:
        self._create_bucket()
        data = bytes(range(256))
        init_plugin(None)
        remote_path = "s3://bucket/path"

        with tempfile.TemporaryDirectory() as tmpdir:
            upload_path = os.path.join(tmpdir, "upload")
            download_path = os.path.join(tmpdir, "download")
            with open(upload_path, "wb") as f:
                f.write(data)

            upload_file(upload_path, remote_path)
            download_file(remote_path, download_path)

            with open(download_path, "rb") as f:
                out_data = f.read()

        self.assertEqual(out_data, data)
