#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import shutil
import tarfile as tar
import tempfile
from typing import Tuple
from urllib.parse import urlparse

import boto3
import botocore
from torchx.runtime.storage import register_storage_provider, StorageProvider


log: logging.Logger = logging.getLogger(__name__)


class S3(StorageProvider):
    SCHEME: str = "s3"

    def __init__(self, session: boto3.Session) -> None:
        self._session = session
        self._s3: botocore.client.BaseClient = session.client("s3")

    def cp(self, target_path: str, bucket: str, key: str) -> str:
        """
        Uploads target_path to s3://bucket/key. If the target_path is a file
        then uploads to s3://bucket/key/file_name, if the target_path is a
        directory, then a tarball is created with the contents of target_path
        and uploaded to s3://bucket/key/dir_name.tar.gz. The tar is created as
        if created by running the command:

        cd target_path && tar xzf /tmp/$(basename target_path).tar.gz *

        Returns the destination s3 url
        """

        target_basename = os.path.basename(target_path)

        if os.path.isdir(target_path):
            tmpdir = tempfile.mkdtemp(prefix="torchx_aws_")
            tar_basename = f"{target_basename}.tar.gz"
            tar_file = os.path.join(tmpdir, tar_basename)
            log.info(f"Compressing {target_path} into {tar_basename}")
            with tar.open(tar_file, "x:gz") as f:
                f.add(target_path, arcname="", recursive=True)

            dest_key = f"{key}/{tar_basename}"
            target_file = tar_file
        else:
            tmpdir = None
            dest_key = f"{key}/{target_basename}"
            target_file = target_path

        log.info(f"Uploading {target_file} to s3://{bucket}/{dest_key}")
        self._s3.upload_file(target_file, bucket, dest_key)

        if tmpdir:
            log.info(f"Deleting tmp dir: {tmpdir}")
            shutil.rmtree(tmpdir)
        return f"s3://{bucket}/{dest_key}"

    def _parse_url(self, url: str) -> Tuple[str, str]:
        parsed = urlparse(url)
        return parsed.netloc, parsed.path[1:]

    def download_file(self, url: str) -> bytes:
        bucket, path = self._parse_url(url)
        return self._s3.get_object(Bucket=bucket, Key=path)["Body"].read()

    def upload_file(self, url: str, body: bytes) -> None:
        bucket, path = self._parse_url(url)
        self._s3.put_object(Bucket=bucket, Key=path, Body=body)


def init_plugin(args: None) -> None:
    register_storage_provider(S3(boto3.Session()))
