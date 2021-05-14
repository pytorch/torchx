#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import os
import tempfile
from contextlib import contextmanager
from typing import Dict
from typing import Generator
from urllib.parse import urlparse


def download_file(url: str) -> bytes:
    return get_storage_provider(url).download_file(url)


def upload_file(url: str, body: bytes) -> None:
    get_storage_provider(url).upload_file(url, body)


class StorageProvider(abc.ABC):
    SCHEME: str

    @abc.abstractmethod
    def download_file(self, url: str) -> bytes:
        ...

    @abc.abstractmethod
    def upload_file(self, url: str, body: bytes) -> None:
        ...


_PROVIDERS: Dict[str, StorageProvider] = {}


def register_storage_provider(provider: StorageProvider) -> None:
    assert provider.SCHEME not in _PROVIDERS
    _PROVIDERS[provider.SCHEME] = provider


def get_storage_provider(url: str) -> StorageProvider:
    parsed = urlparse(url)
    scheme = parsed.scheme
    assert scheme in _PROVIDERS, f"failed to find provider for URL {url}"
    return _PROVIDERS[scheme]


class FileProvider(StorageProvider):
    SCHEME: str = "file"

    def download_file(self, url: str) -> bytes:
        parsed = urlparse(url)
        with open(parsed.path, "rb") as f:
            return f.read()

    def upload_file(self, url: str, body: bytes) -> None:
        parsed = urlparse(url)
        with open(parsed.path, "wb") as f:
            f.write(body)


@contextmanager
def temppath() -> Generator[str, None, None]:
    tf = tempfile.NamedTemporaryFile(delete=False)
    tf.close()
    try:
        yield "file://" + tf.name
    finally:
        os.remove(tf.name)


register_storage_provider(FileProvider())
