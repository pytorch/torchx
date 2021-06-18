# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

try:
    from importlib.metadata import EntryPoint
except ImportError:
    from importlib_metadata import EntryPoint
from configparser import ConfigParser
from typing import Dict
from typing import List
from unittest.mock import MagicMock, patch

from torchx.util.entrypoints import load, load_group


def EntryPoint_from_config(config: ConfigParser) -> List[EntryPoint]:
    # from stdlib, Copyright (c) Python Authors
    return [
        EntryPoint(name, value, group)
        for group in config.sections()
        for name, value in config.items(group)
    ]


def EntryPoint_from_text(text: str) -> List[EntryPoint]:
    # from stdlib, Copyright (c) Python Authors
    config = ConfigParser(delimiters="=")
    config.read_string(text)
    return EntryPoint_from_config(config)


def foobar() -> str:
    return "foobar"


def barbaz() -> str:
    return "barbaz"


_EP_TXT: str = """
[entrypoints.test]
foo = torchx.util.test.entrypoints_test:foobar
"""

_EP_GRP_TXT: str = """
[ep.grp.test]
foo = torchx.util.test.entrypoints_test:foobar
bar = torchx.util.test.entrypoints_test:barbaz
"""

_EP_GRP_IGN_ATTR_TXT: str = """
[ep.grp.missing.attr.test]
baz = torchx.util.test.entrypoints_test:missing_attr
"""

_EP_GRP_IGN_MOD_TXT: str = """
[ep.grp.missing.mod.test]
baz = torchx.util.test.entrypoints_test.missing_module
"""
_ENTRY_POINTS: Dict[str, List[EntryPoint]] = {
    "entrypoints.test": EntryPoint_from_text(_EP_TXT),
    "ep.grp.test": EntryPoint_from_text(_EP_GRP_TXT),
    "ep.grp.missing.attr.test": EntryPoint_from_text(_EP_GRP_IGN_ATTR_TXT),
    "ep.grp.missing.mod.test": EntryPoint_from_text(_EP_GRP_IGN_MOD_TXT),
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

    @patch(_METADATA_EPS, return_value=_ENTRY_POINTS)
    def test_load_group(self, mock_md_eps: MagicMock) -> None:
        eps = load_group("ep.grp.test")
        self.assertEqual(2, len(eps))
        self.assertEqual("foobar", eps["foo"]())
        self.assertEqual("barbaz", eps["bar"]())

        eps = load_group("ep.grp.test.missing")
        self.assertIsNone(eps)

    @patch(_METADATA_EPS, return_value=_ENTRY_POINTS)
    def test_load_group_with_default(self, mock_md_eps: MagicMock) -> None:
        eps = load_group("ep.grp.test", {"foo": barbaz, "bar": foobar})
        self.assertEqual(2, len(eps))
        self.assertEqual("foobar", eps["foo"]())
        self.assertEqual("barbaz", eps["bar"]())

        eps = load_group("ep.grp.test.missing", {"foo": barbaz, "bar": foobar})
        self.assertEqual(2, len(eps))
        self.assertEqual("barbaz", eps["foo"]())
        self.assertEqual("foobar", eps["bar"]())

    @patch(_METADATA_EPS, return_value=_ENTRY_POINTS)
    def test_load_group_ignore_missing(self, mock_md_eps: MagicMock) -> None:
        eps = load_group("ep.grp.missing.attr.test", ignore_missing=True)
        self.assertFalse(eps)

        eps = load_group("ep.grp.missing.mod.test", ignore_missing=True)
        self.assertFalse(eps)

    @patch(_METADATA_EPS, return_value=_ENTRY_POINTS)
    def test_load_group_not_ignore_missing(self, mock_md_eps: MagicMock) -> None:
        with self.assertRaises(AttributeError):
            load_group("ep.grp.missing.attr.test", ignore_missing=False)

        with self.assertRaises(ModuleNotFoundError):
            load_group("ep.grp.missing.mod.test", ignore_missing=False)
