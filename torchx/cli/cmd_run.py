# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import ast
from typing import Type, List, Iterable, Callable

import torchelastic.tsm.driver as tsm
import yaml
from torchx.cli.cmd_base import SubCommand


class UnsupportFeatureError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"Using unsupported feature {name} in config.")


class ConfValidator(ast.NodeVisitor):
    IMPORT_ALLOWLIST: Iterable[str] = (
        "torchx",
        "torchelastic.tsm",
    )

    FEATURE_BLOCKLIST: Iterable[Type[object]] = (
        # statements
        ast.FunctionDef,
        ast.ClassDef,
        ast.Return,
        ast.Delete,
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.If,
        ast.With,
        ast.AsyncWith,
        ast.Raise,
        ast.Try,
        ast.Global,
        ast.Nonlocal,
        # expressions
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )

    def visit(self, node: ast.AST) -> None:
        if node.__class__ in self.FEATURE_BLOCKLIST:
            raise UnsupportFeatureError(node.__class__.__name__)

        super().visit(node)

    def _validate_import_path(self, names: List[ast.alias]) -> None:
        for alias in names:
            if not any(
                alias.name.startswith(prefix) for prefix in self.IMPORT_ALLOWLIST
            ):
                raise ImportError(
                    f"import {alias.name} not in allowed import prefixes {self.IMPORT_ALLOWLIST}"
                )

    def visit_Import(self, node: ast.Import) -> None:
        self._validate_import_path(node.names)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        self._validate_import_path(node.names)


def _get_arg_type(type_name: str) -> Callable[[str], object]:
    TYPES = (int, str, float)
    for t in TYPES:
        if t.__name__ == type_name:
            return t
    raise TypeError(f"unknown argument type {type_name}")


class CmdRun(SubCommand):
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "script",
            type=str,
            help="Name of the main binary, relative to the fbpkg directory",
        )
        subparser.add_argument(
            "script_args",
            nargs=argparse.REMAINDER,
        )

    def run(self, args: argparse.Namespace) -> None:
        script_path = args.script
        with open(script_path, "r") as f:
            body = f.read()

        frontmatter, script = body.split("\n---\n")

        conf = yaml.safe_load(frontmatter)
        script_parser = argparse.ArgumentParser(description=conf.get("description"))
        for arg in conf["arguments"]:
            arg_type = _get_arg_type(arg.get("type", "str"))
            default = arg.get("default")
            if default:
                default = arg_type(default)
            script_parser.add_argument(
                arg["name"],
                help=arg.get("help"),
                type=arg_type,
                default=default,
            )

        node = ast.parse(script)
        validator = ConfValidator()
        validator.visit(node)

        app = None

        def export(_app: tsm.Application) -> None:
            nonlocal app
            app = _app

        scope = {
            "export": export,
            "args": script_parser.parse_args(args.script_args),
        }

        exec(script, scope)  # noqa: P204

        assert app is not None, "config file did not export an app"
        assert isinstance(
            app, tsm.Application
        ), f"config file did not export a tsm.Application {app}"

        raise NotImplementedError("no implementation for run yet")
