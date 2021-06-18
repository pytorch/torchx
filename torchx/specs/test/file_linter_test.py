# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import ast
import os
import unittest
from typing import Dict, List, Optional, cast

from pyre_extensions import none_throws
from torchx.specs.file_linter import get_fn_docstring, parse_fn_docstring, validate


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


def _test_docstring_empty(arg: str) -> "AppDef":
    """ """
    pass


def _test_docstring_func_desc() -> "AppDef":
    """
    Function description
    """
    pass


def _test_docstring_no_args(arg: str) -> "AppDef":
    """
    Test description
    """
    pass


def _test_docstring_correct(arg0: str, arg1: int, arg2: Dict[int, str]) -> "AppDef":
    """Short Test description

    Long funct description

    Args:
        arg0: arg0 desc
        arg1: arg1 desc
        arg2: arg2 desc
    """
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


def current_file_path() -> str:
    return os.path.join(os.path.dirname(__file__), __file__)


class SpecsFileValidatorTest(unittest.TestCase):
    def setUp(self) -> None:
        self._path = current_file_path()
        with open(self._path, "r") as fp:
            source = fp.read()
        self._file_content = source

    def test_validate_docstring_func_desc(self) -> None:
        linter_errors = validate(
            self._file_content, self._path, torchx_function="_test_docstring_func_desc"
        )
        self.assertEqual(0, len(linter_errors))

    def test_validate_no_return(self) -> None:
        linter_errors = validate(
            self._file_content, self._path, torchx_function="_test_fn_no_return"
        )
        self.assertEqual(1, len(linter_errors))
        expected_desc = (
            "Function: _test_fn_no_return missing return annotation or "
            "has unknown annotations. Supported return annotation: AppDef"
        )
        self.assertEqual(expected_desc, linter_errors[0].description)

    def test_validate_incorrect_return(self) -> None:
        linter_errors = validate(
            self._file_content, self._path, torchx_function="_test_fn_return_int"
        )
        self.assertEqual(1, len(linter_errors))
        expected_desc = (
            "Function: _test_fn_return_int has incorrect return annotation, "
            "supported annotation: AppDef"
        )
        self.assertEqual(expected_desc, linter_errors[0].description)

    def test_validate_empty_fn(self) -> None:
        linter_errors = validate(
            self._file_content, self._path, torchx_function="_test_empty_fn"
        )
        self.assertEqual(1, len(linter_errors))
        linter_error = linter_errors[0]
        self.assertEqual("TorchxFunctionValidator", linter_error.name)

        expected_desc = (
            "`_test_empty_fn` is missing a Google Style docstring, please add one. "
            "For more information on the docstring format see: "
            "https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html"
        )
        self.assertEquals(expected_desc, linter_error.description)
        self.assertEqual(18, linter_error.line)

    def test_validate_docstring_empty(self) -> None:
        linter_errors = validate(
            self._file_content, self._path, torchx_function="_test_docstring_empty"
        )
        self.assertEqual(1, len(linter_errors))
        linter_error = linter_errors[0]
        self.assertEqual("TorchxFunctionValidator", linter_error.name)
        expected_desc = (
            "`_test_docstring_empty` is missing a Google Style docstring, please add one. "
            "For more information on the docstring format see: "
            "https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html"
        )
        self.assertEquals(expected_desc, linter_error.description)

    def test_validate_docstring_no_args(self) -> None:
        linter_errors = validate(
            self._file_content, self._path, torchx_function="_test_docstring_no_args"
        )
        self.assertEqual(1, len(linter_errors))
        linter_error = linter_errors[0]
        self.assertEqual("TorchxFunctionValidator", linter_error.name)
        expected_desc = (
            "`_test_docstring_no_args` not all function arguments"
            " are present in the docstring. Missing args: ['arg']"
        )
        self.assertEqual(expected_desc, linter_error.description)

    def test_validate_docstring_correct(self) -> None:
        linter_errors = validate(
            self._file_content, self._path, torchx_function="_test_docstring_correct"
        )
        self.assertEqual(0, len(linter_errors))

    def test_validate_args_no_type_defs(self) -> None:
        linter_errors = validate(
            self._file_content, self._path, torchx_function="_test_args_no_type_defs"
        )
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
            self._file_content,
            self._path,
            torchx_function="_test_args_dict_list_complex_types",
        )
        self.assertEqual(6, len(linter_errors))
        expected_desc = (
            "`_test_args_dict_list_complex_types` not all function arguments"
            " are present in the docstring. Missing args: ['arg4']"
        )
        self.assertEqual(
            expected_desc,
            linter_errors[0].description,
        )
        self.assertEqual(
            "Arg arg0 missing type annotation", linter_errors[1].description
        )
        self.assertEqual(
            "Arg arg1 missing type annotation", linter_errors[2].description
        )
        self.assertEqual(
            "Dict can only have primitive types", linter_errors[3].description
        )
        self.assertEqual(
            "List can only have primitive types", linter_errors[4].description
        )
        self.assertEqual(
            "`_test_args_dict_list_complex_types` allows only Dict, List as complex types.Argument `arg4` has: Optional",
            linter_errors[5].description,
        )

    def _get_function_def(self, function_name: str) -> ast.FunctionDef:
        module: ast.Module = ast.parse(self._file_content)
        for expr in module.body:
            if type(expr) == ast.FunctionDef:
                func_def = cast(ast.FunctionDef, expr)
                if func_def.name == function_name:
                    return func_def
        raise RuntimeError(f"No function found: {function_name}")

    def test_validate_docstring_full(self) -> None:
        func_def = self._get_function_def("_test_docstring_correct")
        docstring = none_throws(ast.get_docstring(func_def))

        func_desc, param_desc = parse_fn_docstring(docstring)
        self.assertEqual("Short Test description", func_desc)
        self.assertEqual("arg0 desc", param_desc["arg0"])
        self.assertEqual("arg1 desc", param_desc["arg1"])
        self.assertEqual("arg2 desc", param_desc["arg2"])

    def test_get_fn_docstring(self) -> None:
        function_desc, _ = none_throws(
            get_fn_docstring(self._file_content, "_test_args_dict_list_complex_types")
        )
        self.assertEqual("Test description", function_desc)

    def test_unknown_function(self) -> None:
        linter_errors = validate(
            self._file_content, self._path, torchx_function="unknown_function"
        )
        self.assertEqual(1, len(linter_errors))
        self.assertEqual(
            "Function unknown_function not found", linter_errors[0].description
        )
