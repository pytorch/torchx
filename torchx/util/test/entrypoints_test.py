# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from importlib.metadata import EntryPoint
from typing import Dict
from unittest.mock import MagicMock, patch

from torchx.util.entrypoints import load


def foobar() -> str:
    return "foobar"


def barbaz() -> str:
    return "barbaz"


_ENTRY_POINT_TXT: str = """
[entrypoints.test]
foo = torchx.util.test.entrypoints_test:foobar
"""

_ENTRY_POINTS: Dict[str, EntryPoint] = {
    # pyre-ignore[16]
    "entrypoints.test": EntryPoint._from_text(_ENTRY_POINT_TXT)
}

_METADATA_EPS: str = "torchx.util.entrypoints.metadata.entry_points"


class EntryPointsTest(unittest.TestCase):
    @patch(_METADATA_EPS, return_value=_ENTRY_POINTS)
    def test_load(self, mock_md_eps: MagicMock) -> None:
        print(type(load("entrypoints.test", "foo")))
        self.assertEqual("foobar", load("entrypoints.test", "foo")())

    @patch(_METADATA_EPS, return_value=_ENTRY_POINTS)
    def test_load_with_default(self, mock_md_eps: MagicMock) -> None:
        self.assertEqual("barbaz", load("entrypoints.test", "missing", barbaz)())
        self.assertEqual("barbaz", load("entrypoints.missing", "foo", barbaz)())
        self.assertEqual("barbaz", load("entrypoints.missing", "missing", barbaz)())
