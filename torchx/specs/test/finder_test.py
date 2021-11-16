#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torchx.specs.finder as finder
from pyre_extensions import none_throws
from torchx.runner import get_runner
from torchx.runtime.tracking import FsspecResultTracker
from torchx.specs.api import AppDef, AppState, Role
from torchx.specs.finder import (
    ComponentNotFoundException,
    ComponentValidationException,
    CustomComponentsFinder,
    ModuleComponentsFinder,
    _load_components,
    get_component,
    get_components,
)


def _test_component(name: str, role_name: str = "worker") -> AppDef:
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


def _test_component_without_docstring(name: str, role_name: str = "worker") -> AppDef:
    return AppDef(
        name, roles=[Role(name=role_name, image="test_image", entrypoint="main.py")]
    )


# pyre-ignore[2]
def invalid_component(name, role_name: str = "worker") -> AppDef:
    return AppDef(
        name, roles=[Role(name=role_name, image="test_image", entrypoint="main.py")]
    )


class DirComponentsFinderTest(unittest.TestCase):
    def test_get_components(self) -> None:
        components = _load_components()
        self.assertTrue(len(components) > 1)
        component = components["utils.echo"]
        self.assertEqual("utils.echo", component.name)
        self.assertEqual(
            "Echos a message to stdout (calls echo)", component.description
        )
        self.assertEqual("echo", component.fn_name)
        self.assertIsNotNone(component.fn)

    def test_get_component_by_name(self) -> None:
        component = none_throws(get_component("utils.echo"))
        self.assertEqual("utils.echo", component.name)
        self.assertEqual("echo", component.fn_name)
        self.assertIsNotNone(component.fn)

    def test_get_invalid_component_by_name(self) -> None:
        test_torchx_group = {"foobar": sys.modules[__name__]}
        finder._components = None
        with patch("torchx.specs.finder.entrypoints") as entrypoints_mock:
            entrypoints_mock.load_group.return_value = test_torchx_group
            with self.assertRaises(ComponentValidationException):
                get_component("foobar.finder_test.invalid_component")

    def test_get_unknown_component_by_name(self) -> None:
        test_torchx_group = {"foobar": sys.modules[__name__]}
        finder._components = None
        with patch("torchx.specs.finder.entrypoints") as entrypoints_mock:
            entrypoints_mock.load_group.return_value = test_torchx_group
            with self.assertRaises(ComponentNotFoundException):
                get_component("foobar.finder_test.unknown_component")

    def test_get_invalid_component(self) -> None:
        test_torchx_group = {"foobar": sys.modules[__name__]}
        with patch("torchx.specs.finder.entrypoints") as entrypoints_mock:
            entrypoints_mock.load_group.return_value = test_torchx_group
            components = _load_components()
        foobar_component = components["foobar.finder_test.invalid_component"]
        self.assertEqual(1, len(foobar_component.validation_errors))

    def test_get_entrypoints_components(self) -> None:
        test_torchx_group = {"foobar": sys.modules[__name__]}
        with patch("torchx.specs.finder.entrypoints") as entrypoints_mock:
            entrypoints_mock.load_group.return_value = test_torchx_group
            components = _load_components()
        foobar_component = components["foobar.finder_test._test_component"]
        self.assertEqual(_test_component, foobar_component.fn)
        self.assertEqual("_test_component", foobar_component.fn_name)
        self.assertEqual("foobar.finder_test._test_component", foobar_component.name)
        self.assertEqual("Test component", foobar_component.description)

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


def current_file_path() -> str:
    return os.path.join(os.path.dirname(__file__), __file__)


class CustomComponentsFinderTest(unittest.TestCase):
    def test_find_components(self) -> None:
        components = CustomComponentsFinder(
            current_file_path(), "_test_component"
        ).find()
        self.assertEqual(1, len(components))
        component = components[0]
        self.assertEqual(f"{current_file_path()}:_test_component", component.name)
        self.assertEqual("Test component", component.description)
        self.assertEqual("_test_component", component.fn_name)
        self.assertListEqual([], component.validation_errors)

    def test_find_components_without_docstring(self) -> None:
        components = CustomComponentsFinder(
            current_file_path(), "_test_component_without_docstring"
        ).find()
        self.assertEqual(1, len(components))
        component = components[0]
        self.assertEqual(
            f"{current_file_path()}:_test_component_without_docstring", component.name
        )
        exprected_desc = """_test_component_without_docstring TIP: improve this help string by adding a docstring
to your component (see: https://pytorch.org/torchx/latest/component_best_practices.html)"""
        self.assertEqual(exprected_desc, component.description)
        self.assertEqual("_test_component_without_docstring", component.fn_name)
        self.assertListEqual([], component.validation_errors)

    def test_get_component(self) -> None:
        component = get_component(f"{current_file_path()}:_test_component")
        self.assertEqual(f"{current_file_path()}:_test_component", component.name)
        self.assertEqual("Test component", component.description)
        self.assertEqual("_test_component", component.fn_name)
        self.assertListEqual([], component.validation_errors)

    def test_get_components(self) -> None:
        components = get_components()
        for component in components.values():
            self.assertListEqual([], component.validation_errors)

    def test_get_component_unknown(self) -> None:
        with self.assertRaises(ComponentNotFoundException):
            get_component(f"{current_file_path()}:unknown_component")

    def test_get_component_invalid(self) -> None:
        with self.assertRaises(ComponentValidationException):
            get_component(f"{current_file_path()}:invalid_component")


class GetBuiltinSourceTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = Path(tempfile.mkdtemp("torchx_specs_finder_test"))

        # this is so that the test can pick up penv python (fb-only)
        # which is added as a test resource
        self.orig_cwd = os.getcwd()
        os.chdir(os.path.dirname(__file__))

    def tearDown(self) -> None:
        os.chdir(self.orig_cwd)
        shutil.rmtree(self.test_dir)

    def test_get_builtin_source_with_echo(self) -> None:
        echo_src = finder.get_builtin_source("utils.echo")

        # save it to a file and try running it
        echo_copy = self.test_dir / "echo_copy.py"
        with open(echo_copy, "w") as f:
            f.write(echo_src)

        runner = get_runner()
        app_handle = runner.run_component(
            scheduler="local_cwd",
            component=f"{str(echo_copy)}:echo",
            component_args=["--msg", "hello world"],
        )
        status = runner.wait(app_handle, wait_interval=0.1)
        self.assertIsNotNone(status)
        self.assertEqual(AppState.SUCCEEDED, status.state)

    def test_get_builtin_source_with_booth(self) -> None:
        # try copying and running a builtin that is NOT the first
        # defined function in the file

        booth_src = finder.get_builtin_source("utils.booth")

        # save it to a file and try running it
        booth_copy = self.test_dir / "booth_copy.py"
        with open(booth_copy, "w") as f:
            f.write(booth_src)

        runner = get_runner()

        trial_idx = 0
        tracker_base = str(self.test_dir / "tracking")

        app_handle = runner.run_component(
            scheduler="local_cwd",
            cfg={"prepend_cwd": True},
            component=f"{str(booth_copy)}:booth",
            component_args=[
                "--x1=1",
                "--x2=3",
                f"--trial_idx={trial_idx}",
                f"--tracker_base={tracker_base}",
            ],
        )
        status = runner.wait(app_handle, wait_interval=0.1)
        self.assertIsNotNone(status)
        self.assertEqual(AppState.SUCCEEDED, status.state)

        tracker = FsspecResultTracker(tracker_base)
        # booth function has global min of 0 at (1, 3)
        self.assertEqual(0, tracker[trial_idx]["booth_eval"])
