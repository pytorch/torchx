#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import ast
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, cast

from docstring_parser import parse
from pyre_extensions import none_throws

# pyre-ignore-all-errors[16]


def get_arg_names(app_specs_func_def: ast.FunctionDef) -> List[str]:
    arg_names = []
    fn_args = app_specs_func_def.args
    for arg_def in fn_args.args:
        arg_names.append(arg_def.arg)
    if fn_args.vararg:
        arg_names.append(fn_args.vararg.arg)
    for arg in fn_args.kwonlyargs:
        arg_names.append(arg.arg)
    return arg_names


def parse_fn_docstring(func_description: str) -> Tuple[str, Dict[str, str]]:
    """
    Given a docstring in a google-style format, returns the function description and
    description of all arguments.
    See: https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
    """
    args_description = {}
    docstring = parse(func_description)
    for param in docstring.params:
        args_description[param.arg_name] = param.description
    short_func_description = docstring.short_description
    return (short_func_description or "", args_description)


def get_fn_docstring(
    source: str, function_name: str
) -> Optional[Tuple[str, Dict[str, str]]]:
    module = ast.parse(source)
    for expr in module.body:
        if type(expr) == ast.FunctionDef:
            func_def = cast(ast.FunctionDef, expr)
            if func_def.name == function_name:
                docstring = ast.get_docstring(func_def)
                if not docstring:
                    return None
                return parse_fn_docstring(docstring)
    return None


def get_short_fn_description(source: str, function_name: str) -> Optional[str]:
    docstring = get_fn_docstring(source, function_name)
    if not docstring:
        return None
    return docstring[0]


@dataclass
class LinterMessage:
    name: str
    description: str
    path: str
    line: int
    char: int
    severity: str = "error"


class TorchxFunctionValidator(abc.ABC):
    def __init__(self, path: str) -> None:
        self._path = path

    @abc.abstractmethod
    def validate(self, app_specs_func_def: ast.FunctionDef) -> List[LinterMessage]:
        raise NotImplementedError()

    def _gen_linter_message(self, description: str, lineno: int) -> LinterMessage:
        return LinterMessage(
            name="TorchxFunctionValidator",
            description=description,
            path=self._path,
            line=lineno,
            char=0,
            severity="error",
        )


class TorchxDocstringValidator(TorchxFunctionValidator):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    def validate(self, app_specs_func_def: ast.FunctionDef) -> List[LinterMessage]:
        """
        Validates the docstring of the `get_app_spec` function. Criteria:
        * There mast be google-style docstring
        * If there are more than zero arguments, there mast be a `Args:` section defined
            with all arguments included.
        """
        docsting = ast.get_docstring(app_specs_func_def)
        lineno = app_specs_func_def.lineno
        if not docsting:
            desc = (
                f"`{app_specs_func_def.name}` is missing a Google Style docstring, please add one. "
                "For more information on the docstring format see: "
                "https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html"
            )
            return [self._gen_linter_message(desc, lineno)]

        arg_names = get_arg_names(app_specs_func_def)
        _, docstring_arg_defs = parse_fn_docstring(docsting)
        missing_args = [
            arg_name for arg_name in arg_names if arg_name not in docstring_arg_defs
        ]
        if len(missing_args) > 0:
            desc = (
                f"`{app_specs_func_def.name}` not all function arguments are present"
                f" in the docstring. Missing args: {missing_args}"
            )
            return [self._gen_linter_message(desc, lineno)]
        return []


class TorchxFunctionArgsValidator(TorchxFunctionValidator):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    def validate(self, app_specs_func_def: ast.FunctionDef) -> List[LinterMessage]:
        linter_errors = []
        for arg_def in app_specs_func_def.args.args:
            arg_linter_errors = self._validate_arg_def(app_specs_func_def.name, arg_def)
            linter_errors += arg_linter_errors
        if app_specs_func_def.args.vararg:
            arg_linter_errors = self._validate_arg_def(
                app_specs_func_def.name, none_throws(app_specs_func_def.args.vararg)
            )
            linter_errors += arg_linter_errors
        for arg in app_specs_func_def.args.kwonlyargs:
            arg_linter_errors = self._validate_arg_def(app_specs_func_def.name, arg)
            linter_errors += arg_linter_errors
        return linter_errors

    def _validate_arg_def(
        self, function_name: str, arg_def: ast.arg
    ) -> List[LinterMessage]:
        if not arg_def.annotation:
            return [
                self._gen_linter_message(
                    f"Arg {arg_def.arg} missing type annotation", arg_def.lineno
                )
            ]
        if isinstance(arg_def.annotation, ast.Name):
            # TODO(aivanou): add support for primitive type check
            return []
        complex_type_def = cast(ast.Subscript, none_throws(arg_def.annotation))
        if complex_type_def.value.id == "Optional":
            # ast module in python3.9 does not have ast.Index wrapper
            if isinstance(complex_type_def.slice, ast.Index):
                complex_type_def = complex_type_def.slice.value
            else:
                complex_type_def = complex_type_def.slice
            # Check if type is Optional[primitive_type]
            if isinstance(complex_type_def, ast.Name):
                return []
        # Check if type is Union[Dict,List]
        type_name = complex_type_def.value.id
        if type_name != "Dict" and type_name != "List":
            desc = (
                f"`{function_name}` allows only Dict, List as complex types."
                f"Argument `{arg_def.arg}` has: {type_name}"
            )
            return [self._gen_linter_message(desc, arg_def.lineno)]
        linter_errors = []
        # ast module in python3.9 does not have objects wrapped in ast.Index
        if isinstance(complex_type_def.slice, ast.Index):
            sub_type = complex_type_def.slice.value
        else:
            sub_type = complex_type_def.slice
        if type_name == "Dict":
            sub_type_tuple = cast(ast.Tuple, sub_type)
            for el in sub_type_tuple.elts:
                if not isinstance(el, ast.Name):
                    desc = "Dict can only have primitive types"
                    linter_errors.append(self._gen_linter_message(desc, arg_def.lineno))
        elif not isinstance(sub_type, ast.Name):
            desc = "List can only have primitive types"
            linter_errors.append(self._gen_linter_message(desc, arg_def.lineno))
        return linter_errors


class TorchxReturnValidator(TorchxFunctionValidator):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    def _get_return_annotation(
        self, app_specs_func_def: ast.FunctionDef
    ) -> Optional[str]:
        return_def = app_specs_func_def.returns
        if not return_def:
            return None
        if isinstance(return_def, ast.Attribute):
            return return_def.attr
        elif isinstance(return_def, ast.Name):
            return return_def.id
        elif isinstance(return_def, ast.Str):
            return return_def.s
        elif isinstance(return_def, ast.Constant):
            return return_def.value
        else:
            return None

    def validate(self, app_specs_func_def: ast.FunctionDef) -> List[LinterMessage]:
        """
        Validates return annotation of the torchx function. Current allowed annotations:
            * AppDef
            * specs.AppDef
        """
        supported_return_annotation = "AppDef"
        return_annotation = self._get_return_annotation(app_specs_func_def)
        linter_errors = []
        if not return_annotation:
            desc = (
                f"Function: {app_specs_func_def.name} missing return annotation "
                f"or has unknown annotations. Supported return annotation: {supported_return_annotation}"
            )
            linter_errors.append(
                self._gen_linter_message(desc, app_specs_func_def.lineno)
            )
        elif return_annotation != supported_return_annotation:
            desc = (
                f"Function: {app_specs_func_def.name} has incorrect return annotation, "
                f"supported annotation: {supported_return_annotation}"
            )
            linter_errors.append(
                self._gen_linter_message(desc, app_specs_func_def.lineno)
            )

        return linter_errors


class TorchFunctionVisitor(ast.NodeVisitor):
    """
    Visitor that finds the torchx_function and runs registered validators on it.
    Current registered validators:

    * TorchxDocstringValidator - validates the docstring of the function.
        Criteria:
          * There format should be google-python
          * If there are more than zero arguments defined, there
            should be obligatory `Args:` section that describes each argument on a new line.

    * TorchxFunctionArgsValidator - validates arguments of the function.
        Criteria:
          * Each argument should be annotated with the type
          * The following types are supported:
                - primitive_types: {int, str, float},
                - Optional[primitive_types],
                - Dict[primitive_types, primitive_types],
                - List[primitive_types],
                - Optional[Dict[primitive_types, primitive_types]],
                - Optional[List[primitive_types]]

    """

    def __init__(self, path: str, torchx_function_name: str) -> None:
        self.validators = [
            TorchxDocstringValidator(path),
            TorchxFunctionArgsValidator(path),
            TorchxReturnValidator(path),
        ]
        self.linter_errors: List[LinterMessage] = []
        self.torchx_function_name = torchx_function_name
        self.visited_function = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        if node.name != self.torchx_function_name:
            return
        self.visited_function = True
        for validatior in self.validators:
            self.linter_errors += validatior.validate(node)


def validate(
    source: str, path: str = "<NONE>", torchx_function: str = "get_app_spec"
) -> List[LinterMessage]:
    try:
        module = ast.parse(source)
    except SyntaxError as ex:
        linter_message = LinterMessage(
            name="TorchxValidator",
            description=ex.msg,
            path=path,
            line=ex.lineno or 0,
            char=ex.offset or 0,
            severity="error",
        )
        return [linter_message]
    visitor = TorchFunctionVisitor(path, torchx_function)
    visitor.visit(module)
    linter_errors = visitor.linter_errors
    if not visitor.visited_function:
        linter_errors.append(
            LinterMessage(
                name="TorchxValidator",
                description=f"Function {torchx_function} not found",
                path=path,
                line=0,
                char=0,
                severity="error",
            )
        )
    return linter_errors
