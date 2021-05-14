# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import boto3
from moto import mock_s3
from torchx.aws.s3 import init_plugin
from torchx.runtime.storage import download_file, upload_file


class S3Test(unittest.TestCase):
    @mock_s3
    def test_storage_provider(self) -> None:
        client = boto3.Session().client("s3")
        client.create_bucket(Bucket="bucket")

        init_plugin(None)
        path = "s3://bucket/path"
        body = b"foo"
        upload_file(path, body)
        self.assertEqual(download_file(path), body)
