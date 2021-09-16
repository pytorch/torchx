#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import posixpath
from enum import Enum


class ImageType(str, Enum):
    UNKNOWN = "unknown"
    DIR = "dir"
    DOCKER = "docker"


def image_type(image: str) -> ImageType:
    if not image:
        return ImageType.UNKNOWN
    if posixpath.isabs(image):
        return ImageType.DIR
    return ImageType.DOCKER


def assert_image_type(image: str, expected: ImageType) -> None:
    actual = image_type(image)
    if expected != actual:
        raise TypeError(
            f"expected image of type {repr(expected)} not {repr(actual)}: {repr(image)}",
        )
