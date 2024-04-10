#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import abc
import argparse
import ast
import inspect
from dataclasses import dataclass
from typing import Callable, cast, Dict, List, Optional, Tuple

from docstring_parser import parse
from torchx.util.io import read_conf_file
from torchx.util.types import none_throws


# pyre-ignore-all-errors[16]


def _get_default_arguments_descriptions(fn: Callable[..., object]) -> Dict[str, str]:
    parameters = inspect.signature(fn).parameters
    args_decs = {}
    for parameter_name in parameters.keys():
        # The None or Empty string values getting ignored during help command by argparse
        args_decs[parameter_name] = " "
    return args_decs


class TorchXArgumentHelpFormatter(
    argparse.RawDescriptionHelpFormatter,
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.MetavarTypeHelpFormatter,
):
    """Help message formatter which adds default values and required to argument help.

    If the argument is required, the class appends `(required)` at the end of the help message.
    If the argument has default value, the class appends `(default: $DEFAULT)` at the end.
    The formatter is designed to be used only for the torchx components functions.
    These functions do not have both required and default arguments.
    """

    def _get_help_string(self, action: argparse.Action) -> str:
        help = action.help or ""
        # Only `--help` will have be SUPPRESS, so we ignore it
        if action.default is argparse.SUPPRESS:
            return help
        if action.required:
            help += " (required)"
        else:
            help += f" (default: {action.default})"
        return help


def get_fn_docstring(fn: Callable[..., object]) -> Tuple[str, Dict[str, str]]:
    """
    Parses the function and arguments description from the provided function. Docstring should be in
    `google-style format <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_

    If function has no docstring, the function description will be the name of the function, TIP
    on how to improve the help message and arguments descriptions will be names of the arguments.

    The arguments that are not present in the docstring will contain default/required information

    Args:
        fn: Function with or without docstring

    Returns:
        function description, arguments description where key is the name of the argument and value
            if the description
    """
    default_fn_desc = f"""{fn.__name__} TIP: improve this help string by adding a docstring
to your component (see: https://pytorch.org/torchx/latest/component_best_practices.html)"""
    args_description = _get_default_arguments_descriptions(fn)
    func_description = inspect.getdoc(fn)
    if not func_description:
        return default_fn_desc, args_description
    docstring = parse(func_description)
    for param in docstring.params:
        if param.description is not None:
            args_description[param.arg_name] = param.description
    short_func_description = docstring.short_description or default_fn_desc
    if docstring.long_description:
        short_func_description += "\n" + docstring.long_description
    return (short_func_description or default_fn_desc, args_description)


@dataclass
class LinterMessage:
    name: str
    description: str
    line: int
    char: int
    severity: str = "error"


class TorchxFunctionValidator(abc.ABC):
    @abc.abstractmethod
    def validate(self, app_specs_func_def: ast.FunctionDef) -> List[LinterMessage]:
        """
        Method to call to validate the provided function def.
        """
        raise NotImplementedError()

    def _gen_linter_message(self, description: str, lineno: int) -> LinterMessage:
        return LinterMessage(
            name="TorchxFunctionValidator",
            description=description,
            line=lineno,
            char=0,
            severity="error",
        )


class TorchxFunctionArgsValidator(TorchxFunctionValidator):
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
    Visitor that finds the component_function and runs registered validators on it.
    Current registered validators:

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

    def __init__(self, component_function_name: str) -> None:
        self.validators = [
            TorchxFunctionArgsValidator(),
            TorchxReturnValidator(),
        ]
        self.linter_errors: List[LinterMessage] = []
        self.component_function_name = component_function_name
        self.visited_function = False

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """
        Validates the function def with the child validators.
        """
        if node.name != self.component_function_name:
            return
        self.visited_function = True
        for validator in self.validators:
            self.linter_errors += validator.validate(node)


def validate(path: str, component_function: str) -> List[LinterMessage]:
    """
    Validates the function to make sure it complies the component standard.

    ``validate`` finds the ``component_function`` and vaidates it for according to the following rules:

    #. The function must have `google-styple docs <https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html>`_
    #. All function parameters must be annotated
    #. The function must return :py:class:`torchx.specs.api.AppDef`

    Args:
        path: Path to python source file.
        component_function: Name of the function to be validated.

    Returns:
        List[LinterMessage]: List of validation errors
    """
    source = read_conf_file(path)
    try:
        module = ast.parse(source)
    except SyntaxError as ex:
        linter_message = LinterMessage(
            name="TorchXValidator",
            description=ex.msg,
            line=ex.lineno or 0,
            char=ex.offset or 0,
            severity="error",
        )
        return [linter_message]
    visitor = TorchFunctionVisitor(component_function)
    visitor.visit(module)
    linter_errors = visitor.linter_errors
    if not visitor.visited_function:
        linter_errors.append(
            LinterMessage(
                name="TorchXValidator",
                description=f"Function {component_function} not found",
                line=0,
                char=0,
                severity="error",
            )
        )
    return linter_errors
