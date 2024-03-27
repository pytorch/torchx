# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import s3fs


class MinioFS(s3fs.S3FileSystem):
    """
    A test FS that uses a MinIO filesystem on top of s3fs for TorchX integration
    tests in minikube.
    """

    # pyre-ignore[15] declared in fsspec.spec.AbstractFileSystem.protocol as ClassVar[str | tuple[str, ...]]
    protocol = ("torchx_minio", "s3", "s3a")

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(
            *args,
            key="minio",
            secret="minio123",
            client_kwargs={
                "endpoint_url": "http://minio-service:9000",
            },
            **kwargs,
        )
