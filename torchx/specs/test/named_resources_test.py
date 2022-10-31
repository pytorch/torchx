#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import unittest
from unittest import mock

from torchx.specs import _NamedResourcesLibrary, Resource


def mock_resource() -> Resource:
    return Resource(cpu=0, gpu=0, memMB=0)


class NamedResourcesTest(unittest.TestCase):
    @mock.patch("torchx.specs._named_resource_factories")
    def test_named_resources_library(
        self, mock_named_resources: mock.MagicMock
    ) -> None:
        mock_named_resources.keys.return_value = [
            "p3.2xlarge",
            "p3.16xlarge",
            "p4d.24xlarge",
        ]

        with self.assertRaisesRegex(
            KeyError,
            "No named resource found for `foo`. Registered named resources:.*",
        ):
            _ = _NamedResourcesLibrary()["foo"]

        with self.assertRaisesRegex(
            KeyError,
            "No named resource found for `p316xl`. Did you mean `p3.16xlarge`?",
        ):
            _ = _NamedResourcesLibrary()["p316xl"]
