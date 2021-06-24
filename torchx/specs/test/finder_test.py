#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import unittest
from unittest.mock import patch

from pyre_extensions import none_throws
from torchx.specs.api import AppDef, Role
from torchx.specs.finder import (
    ModuleComponentsFinder,
    get_component,
    _load_components,
)


def test_component(name: str, role_name: str = "worker") -> AppDef:
    """
    Test component

    Args:
        name: AppDef name
        role_name: Role name

    Returns:
        AppDef
    """
    return AppDef(
        name, roles=[Role(name=role_name, image="test_image", entrypoint="main.py")]
    )


def invalid_component(name: str, role_name: str = "worker") -> AppDef:
    return AppDef(
        name, roles=[Role(name=role_name, image="test_image", entrypoint="main.py")]
    )


class DirComponentsFinderTest(unittest.TestCase):
    def test_get_components(self) -> None:
        components = _load_components()
        self.assertTrue(len(components) > 1)
        component = components["utils.echo"]
        self.assertEqual("torchx.components.utils", component.module_name)
        self.assertEqual("utils.echo", component.name)
        self.assertEqual(
            "Echos a message to stdout (calls /bin/echo)", component.description
        )
        self.assertEqual("echo", component.fn_name)
        self.assertEqual("", component.group)
        self.assertIsNotNone(component.fn)

    def test_get_component_by_name(self) -> None:
        component = none_throws(get_component("utils.echo"))
        self.assertEqual("utils.echo", component.name)
        self.assertEqual("torchx.components.utils", component.module_name)
        self.assertEqual("echo", component.fn_name)
        self.assertIsNotNone(component.fn)

    def test_get_entrypoints_components(self) -> None:
        test_torchx_group = {"foobar": sys.modules[__name__]}
        with patch("torchx.specs.finder.entrypoints") as entrypoints_mock:
            entrypoints_mock.load_group.return_value = test_torchx_group
            components = _load_components()
        foobar_component = components["foobar.finder_test.test_component"]
        self.assertEqual(test_component, foobar_component.fn)
        self.assertEqual("test_component", foobar_component.fn_name)
        self.assertEqual("foobar.finder_test.test_component", foobar_component.name)
        self.assertEqual("torchx.specs.test.finder_test", foobar_component.module_name)
        self.assertEqual("Test component", foobar_component.description)

    def test_validate_and_get_description(self) -> None:
        expected_desc = "Test component"
        finder = ModuleComponentsFinder(sys.modules[__name__], "")
        actual_desc = finder._validate_and_get_description(
            sys.modules[__name__], "test_component"
        )
        self.assertEqual(expected_desc, actual_desc)

    def test_validate_and_get_description_invalid_component(self) -> None:
        finder = ModuleComponentsFinder(sys.modules[__name__], "")
        actual_desc = finder._validate_and_get_description(
            sys.modules[__name__], "invalid_component"
        )
        self.assertIsNone(actual_desc)

    def test_get_base_module_name(self) -> None:
        finder = ModuleComponentsFinder(sys.modules[__name__], "")
        expected_name = "torchx.specs.test"
        actual_name = finder._get_base_module_name(sys.modules[__name__])
        self.assertEqual(expected_name, actual_name)

    def test_get_base_module_name_for_init_module(self) -> None:
        finder = ModuleComponentsFinder("", "")
        expected_name = "torchx.specs"
        actual_name = finder._get_base_module_name(sys.modules["torchx.specs"])
        self.assertEqual(expected_name, actual_name)

    def test_get_component_name(self) -> None:
        finder = ModuleComponentsFinder("", group="foobar")
        actual_name = finder._get_component_name(
            "test.main_module", "test.main_module.sub_module.bar", "get_component"
        )
        expected_name = "foobar.sub_module.bar.get_component"
        self.assertEqual(expected_name, actual_name)

    def test_strip_init(self) -> None:
        finder = ModuleComponentsFinder("", "")
        self.assertEqual("foobar", finder._strip_init("foobar.__init__"))
        self.assertEqual("", finder._strip_init("__init__"))
        self.assertEqual("foobar", finder._strip_init("foobar"))

    def test_get_module_name(self) -> None:
        finder = ModuleComponentsFinder("", "")
        actual_name = finder._get_module_name(
            "/test/path/main_module/foobar.py", "/test/path", "main"
        )
        expected_name = "main.main_module.foobar"
        self.assertEqual(expected_name, actual_name)
