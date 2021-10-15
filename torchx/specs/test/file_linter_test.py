# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import unittest
from typing import Dict, List, Optional
from unittest.mock import patch

from torchx.specs.file_linter import (
    get_fn_docstring,
    validate,
    TorchXArgumentHelpFormatter,
)


# Note if the function is moved, the tests need to be updated with new lineno
# pyre-ignore[11]: Ignore unknown type "AppDef"
def _test_empty_fn() -> "AppDef":
    pass


# Note if the function is moved, the tests need to be updated with new lineno
# pyre-ignore[3]: Omit return value for testing purposes
def _test_fn_no_return():
    """
    Function description
    """
    pass


def _test_fn_return_int() -> int:
    """
    Function description
    """
    return 0


def _test_docstring(arg0: str, arg1: int, arg2: Dict[int, str]) -> "AppDef":
    """Short Test description

    Long funct description

    Args:
        arg0: arg0 desc
        arg1: arg1 desc
    """
    pass


def _test_docstring_short() -> "AppDef":
    """Short Test description"""
    pass


def _test_without_docstring(arg0: str) -> "AppDef":
    pass


# pyre-ignore[2]: Omit return value for testing purposes
def _test_args_no_type_defs(arg0, arg1, arg2: Dict[int, str]) -> "AppDef":
    """
    Test description

    Args:
        arg0: arg0 desc
        arg1: arg1 desc
        arg2: arg2 desc
    """
    pass


def _test_args_dict_list_complex_types(
    # pyre-ignore[2]: Omit return value for testing purposes
    arg0,
    # pyre-ignore[2]: Omit return value for testing purposes
    arg1,
    arg2: Dict[int, List[str]],
    arg3: List[List[str]],
    arg4: Optional[Optional[str]],
) -> "AppDef":
    """
    Test description

    Args:
        arg0: arg0 desc
        arg1: arg1 desc
        arg2: arg2 desc
        arg3: arg2 desc
    """
    pass


# pyre-ignore[2]
def _test_invalid_fn_with_varags_and_kwargs(*args, id: int) -> "AppDef":
    """
    Test description

    Args:
        args: args desc
    """
    pass


def current_file_path() -> str:
    return os.path.join(os.path.dirname(__file__), __file__)


class SpecsFileValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._path = current_file_path()
        with open(self._path, "r") as fp:
            source = fp.read()
        self._file_content = source

    def test_syntax_error(self) -> None:
        content = "!!foo====bar"
        with patch("torchx.specs.file_linter.read_conf_file") as read_conf_file_mock:
            read_conf_file_mock.return_value = content
            errors = validate(self._path, "unknown_function")
            self.assertEqual(1, len(errors))
            self.assertEqual("invalid syntax", errors[0].description)

    def test_validate_varargs_kwargs_fn(self) -> None:
        linter_errors = validate(
            self._path,
            "_test_invalid_fn_with_varags_and_kwargs",
        )
        self.assertEqual(1, len(linter_errors))
        self.assertTrue(
            "Arg args missing type annotation", linter_errors[0].description
        )

    def test_validate_no_return(self) -> None:
        linter_errors = validate(self._path, "_test_fn_no_return")
        self.assertEqual(1, len(linter_errors))
        expected_desc = (
            "Function: _test_fn_no_return missing return annotation or "
            "has unknown annotations. Supported return annotation: AppDef"
        )
        self.assertEqual(expected_desc, linter_errors[0].description)

    def test_validate_incorrect_return(self) -> None:
        linter_errors = validate(self._path, "_test_fn_return_int")
        self.assertEqual(1, len(linter_errors))
        expected_desc = (
            "Function: _test_fn_return_int has incorrect return annotation, "
            "supported annotation: AppDef"
        )
        self.assertEqual(expected_desc, linter_errors[0].description)

    def test_validate_empty_fn(self) -> None:
        linter_errors = validate(self._path, "_test_empty_fn")
        self.assertEqual(0, len(linter_errors))

    def test_validate_args_no_type_defs(self) -> None:
        linter_errors = validate(self._path, "_test_args_no_type_defs")
        print(linter_errors)
        self.assertEqual(2, len(linter_errors))
        self.assertEqual(
            "Arg arg0 missing type annotation", linter_errors[0].description
        )
        self.assertEqual(
            "Arg arg1 missing type annotation", linter_errors[1].description
        )

    def test_validate_args_no_type_defs_complex(self) -> None:
        linter_errors = validate(
            self._path,
            "_test_args_dict_list_complex_types",
        )
        self.assertEqual(5, len(linter_errors))
        self.assertEqual(
            "Arg arg0 missing type annotation", linter_errors[0].description
        )
        self.assertEqual(
            "Arg arg1 missing type annotation", linter_errors[1].description
        )
        self.assertEqual(
            "Dict can only have primitive types", linter_errors[2].description
        )
        self.assertEqual(
            "List can only have primitive types", linter_errors[3].description
        )
        self.assertEqual(
            "`_test_args_dict_list_complex_types` allows only Dict, List as complex types.Argument `arg4` has: Optional",
            linter_errors[4].description,
        )

    def test_validate_docstring(self) -> None:
        func_desc, param_desc = get_fn_docstring(_test_docstring)
        self.assertEqual("Short Test description ...", func_desc)
        self.assertEqual("arg0 desc", param_desc["arg0"])
        self.assertEqual("arg1 desc", param_desc["arg1"])
        self.assertEqual(" ", param_desc["arg2"])

    def test_validate_docstring_short(self) -> None:
        func_desc, param_desc = get_fn_docstring(_test_docstring_short)
        self.assertEqual("Short Test description", func_desc)

    def test_validate_docstring_no_docs(self) -> None:
        func_desc, param_desc = get_fn_docstring(_test_without_docstring)
        expected_fn_desc = """_test_without_docstring TIP: improve this help string by adding a docstring
to your component (see: https://pytorch.org/torchx/latest/component_best_practices.html)"""
        self.assertEqual(expected_fn_desc, func_desc)
        self.assertEqual(" ", param_desc["arg0"])

    def test_validate_unknown_function(self) -> None:
        linter_errors = validate(self._path, "unknown_function")
        self.assertEqual(1, len(linter_errors))
        self.assertEqual(
            "Function unknown_function not found", linter_errors[0].description
        )

    def test_formatter(self) -> None:
        parser = argparse.ArgumentParser(
            prog="test prog",
            description="test desc",
        )
        parser.add_argument(
            "--foo",
            type=int,
            required=True,
            help="foo",
        )
        parser.add_argument(
            "--bar",
            type=int,
            help="bar",
            default=1,
        )
        formatter = TorchXArgumentHelpFormatter(prog="test")
        self.assertEqual(
            "show this help message and exit",
            formatter._get_help_string(parser._actions[0]),
        )
        self.assertEqual(
            "foo (required)", formatter._get_help_string(parser._actions[1])
        )
        self.assertEqual(
            "bar (default: 1)", formatter._get_help_string(parser._actions[2])
        )
