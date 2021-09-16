#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest

from torchx.schedulers.images import (
    image_type,
    assert_image_type,
    ImageType,
)


class TorchxImageTypeTest(unittest.TestCase):
    def test_image_type(self) -> None:
        self.assertEqual(image_type("foo"), ImageType.DOCKER)
        self.assertEqual(image_type("foo:bar"), ImageType.DOCKER)
        self.assertEqual(image_type("abc/foo-bar:v1.0.0"), ImageType.DOCKER)
        self.assertEqual(image_type("pytorch.com/abc/foo-bar:v1.0.0"), ImageType.DOCKER)
        self.assertEqual(image_type("foo/bar"), ImageType.DOCKER)

        self.assertEqual(image_type("/tmp"), ImageType.DIR)
        self.assertEqual(image_type("/home/test"), ImageType.DIR)
        self.assertEqual(image_type("/"), ImageType.DIR)

        self.assertEqual(image_type(""), ImageType.UNKNOWN)

    def test_assert_image_type(self) -> None:
        with self.assertRaisesRegex(
            TypeError, "expected image of type.*DIR.*not.*DOCKER"
        ):
            assert_image_type("foo/bar", ImageType.DIR)

        with self.assertRaisesRegex(
            TypeError, "expected image of type.*DIR.*not.*UNKNOWN"
        ):
            assert_image_type("", ImageType.DIR)

        with self.assertRaisesRegex(
            TypeError, "expected image of type.*DOCKER.*not.*DIR"
        ):
            assert_image_type("/", ImageType.DOCKER)
