#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import os
import shutil
import tempfile
from contextlib import contextmanager
from typing import Dict
from typing import Generator
from urllib.parse import urlparse


def download_blob(url: str) -> bytes:
    return get_storage_provider(url).download_blob(url)


def upload_blob(url: str, body: bytes) -> None:
    get_storage_provider(url).upload_blob(url, body)


def download_file(url: str, path: str) -> None:
    return get_storage_provider(url).download_file(url, path)


def upload_file(path: str, url: str) -> None:
    get_storage_provider(url).upload_file(path, url)


class StorageProvider(abc.ABC):
    SCHEME: str

    @abc.abstractmethod
    def download_blob(self, url: str) -> bytes:
        ...

    @abc.abstractmethod
    def upload_blob(self, url: str, body: bytes) -> None:
        ...

    @abc.abstractmethod
    def download_file(self, url: str, path: str) -> None:
        ...

    @abc.abstractmethod
    def upload_file(self, path: str, url: str) -> None:
        ...


_PROVIDERS: Dict[str, StorageProvider] = {}


def register_storage_provider(provider: StorageProvider) -> None:
    assert provider.SCHEME not in _PROVIDERS
    _PROVIDERS[provider.SCHEME] = provider


def get_storage_provider(url: str) -> StorageProvider:
    parsed = urlparse(url)
    scheme = parsed.scheme
    assert (
        scheme in _PROVIDERS
    ), f"failed to find provider {scheme} for URL {url} - must be one of {list(_PROVIDERS.keys())}"
    return _PROVIDERS[scheme]


class FileProvider(StorageProvider):
    SCHEME: str = "file"

    def download_blob(self, url: str) -> bytes:
        """
        download_blob fetches the contents of the file located at the URL.
        """
        parsed = urlparse(url)
        with open(parsed.path, "rb") as f:
            return f.read()

    def upload_blob(self, url: str, body: bytes) -> None:
        """
        upload_blob uploads the body to the location specified by the URL.
        """
        parsed = urlparse(url)
        with open(parsed.path, "wb") as f:
            f.write(body)

    def download_file(self, url: str, path: str) -> None:
        """
        download_file downloads the file located at the URL to a location on disk.
        """
        parsed = urlparse(url)
        shutil.copyfile(parsed.path, path)

    def upload_file(self, path: str, url: str) -> None:
        """
        upload_file uploads a file on disk to the location specified by the URL.
        """
        parsed = urlparse(url)
        shutil.copyfile(path, parsed.path)


@contextmanager
def temppath() -> Generator[str, None, None]:
    tf = tempfile.NamedTemporaryFile(delete=False)
    tf.close()
    try:
        yield "file://" + tf.name
    finally:
        os.remove(tf.name)


register_storage_provider(FileProvider())
