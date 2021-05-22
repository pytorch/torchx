# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import ast
import os
import warnings
from dataclasses import asdict
from os import path
from pathlib import Path
from pprint import pformat
from typing import Callable, Iterable, List, Optional, Type

import torchx.specs as specs
import yaml
from torchx.cli.cmd_base import SubCommand
from torchx.cli.conf_helpers import parse_args_children
from torchx.runner import get_runner
from torchx.util import entrypoints


class UnsupportFeatureError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"Using unsupported feature {name} in config.")


class ConfValidator(ast.NodeVisitor):
    IMPORT_ALLOWLIST: Iterable[str] = (
        "torchx",
        "torchelastic.tsm",
        "os.path",
        "pytorch.elastic.torchelastic.tsm",
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
        # ast.GeneratorExp,
    )

    def visit(self, node: ast.AST) -> None:
        if node.__class__ in self.FEATURE_BLOCKLIST:
            raise UnsupportFeatureError(node.__class__.__name__)

        super().visit(node)

    def _validate_import_path(self, names: List[str]) -> None:
        for name in names:
            if not any(name.startswith(prefix) for prefix in self.IMPORT_ALLOWLIST):
                raise ImportError(
                    f"import {name} not in allowed import prefixes {self.IMPORT_ALLOWLIST}"
                )

    def visit_Import(self, node: ast.Import) -> None:
        self._validate_import_path([alias.name for alias in node.names])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if module := node.module:
            self._validate_import_path([module])


def _get_arg_type(type_name: str) -> Callable[[str], object]:
    TYPES = (int, str, float)
    for t in TYPES:
        if t.__name__ == type_name:
            return t
    raise TypeError(f"unknown argument type {type_name}")


def _parse_run_config(arg: str) -> specs.RunConfig:
    conf = specs.RunConfig()
    for key, value in parse_args_children(arg).items():
        conf.set(key, value)
    return conf


# TODO kiuk@ move read_conf_file + _builtins to the Runner once the Runner is API stable

_CONFIG_DIR: Path = Path("torchx/cli/config")
_CONFIG_EXT = ".torchx"


def get_file_contents(conf_file: str) -> Optional[str]:
    """
    Reads the ``conf_file`` relative to the root of the project.
    Returns ``None`` if ``$root/$conf_file`` does not exist.
    Example: ``get_file("torchx/cli/config/foo.txt")``
    """

    module = __name__.replace(".", path.sep)  # torchx/cli/cmd_run
    module_path, _ = path.splitext(__file__)  # $root/torchx/cli/cmd_run
    root = module_path.replace(module, "")
    abspath = path.join(root, conf_file)

    if path.exists(abspath):
        with open(abspath, "r") as f:
            return f.read()
    else:
        return None


def read_conf_file(conf_file: str) -> str:
    builtin_conf = entrypoints.load(
        "torchx.file",
        "get_file_contents",
        default=get_file_contents,
    )(str(_CONFIG_DIR / conf_file))

    # user provided conf file precedes the builtin config
    # just print a warning but use the user provided one
    if path.exists(conf_file):
        if builtin_conf:
            warnings.warn(
                f"The provided config file: {conf_file} overlaps"
                f" with a built-in. It is recommended that you either"
                f" rename the config file or use abs path."
                f" Will use: {path.abspath(conf_file)} for this run."
            )
        with open(conf_file, "r") as f:
            return f.read()
    elif builtin_conf:  # conf_file does not exist fallback to builtin
        return builtin_conf
    else:  # neither conf_file nor builtin exists, raise error
        raise FileNotFoundError(
            f"{conf_file} does not exist and is not a builtin."
            " For a list of available builtins run `torchx builtins`"
        )


def _builtins() -> List[str]:
    builtins: List[str] = []
    for f in os.listdir(_CONFIG_DIR):
        _, extension = os.path.splitext(f)
        if f.endswith(_CONFIG_EXT):
            builtins.append(f)

    return builtins


class CmdBuiltins(SubCommand):
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        pass  # no arguments

    def run(self, args: argparse.Namespace) -> None:
        builtin_configs = _builtins()
        num_builtins = len(builtin_configs)
        print(f"Found {num_builtins} builtin configs:")
        for i, name in enumerate(builtin_configs):
            print(f" {i + 1:2d}. {name}")


class CmdRun(SubCommand):
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--scheduler",
            type=str,
            help="Name of the scheduler to use",
        )
        subparser.add_argument(
            "--scheduler_args",
            type=_parse_run_config,
            help="Arguments to pass to the scheduler (Ex:`cluster=foo,user=bar`)."
            " For a list of scheduler run options run: `torchx runopts`"
            "",
        )
        subparser.add_argument(
            "--dryrun",
            action="store_true",
            default=False,
            help="Does not actually submit the app,"
            " just prints the scheduler request",
        )
        subparser.add_argument(
            "--verbose",
            action="store_true",
            default=False,
            help="Verbose mode, pretty print the app spec",
        )
        subparser.add_argument(
            "conf_file",
            type=str,
            help="Name of builtin conf or path of the *.torchx.conf file."
            " for a list of available builtins run:`torchx builtins`",
        )
        subparser.add_argument(
            "conf_args",
            nargs=argparse.REMAINDER,
        )

    def run(self, args: argparse.Namespace) -> None:
        body = read_conf_file(args.conf_file)
        node = ast.parse(body)

        # we expect the docstring of the conf file to be yaml
        docstring = ast.get_docstring(node)
        if not docstring:
            raise RuntimeError(
                f"Missing parameters docstring in conf file: {args.conf_file}."
                f" Please reference `torchx.cli.config.simple_example.torchx` as an example"
            )
        conf = yaml.safe_load(docstring)
        script_parser = argparse.ArgumentParser(
            prog=f"torchx run {args.conf_file}", description=conf.get("description")
        )
        for arg in conf["arguments"]:
            arg_type = _get_arg_type(arg.get("type", "str"))
            default = arg.get("default")
            if default:
                default = arg_type(default)
            script_args = {
                "help": arg.get("help"),
                "type": arg_type,
                "default": default,
            }
            if arg.get("remainder"):
                script_args["nargs"] = argparse.REMAINDER

            script_parser.add_argument(
                arg["name"],
                **script_args,
            )

        validator = ConfValidator()
        validator.visit(node)

        app = None

        def export(_app: specs.Application) -> None:
            nonlocal app
            app = _app

        scope = {
            "export": export,
            "args": script_parser.parse_args(args.conf_args),
            "scheduler": args.scheduler,
        }

        exec(body, scope)  # noqa: P204

        assert app is not None, "config file did not export an app"
        assert isinstance(
            app, specs.Application
        ), f"config file did not export a torchx.spec.Application {app}"

        runner = get_runner()

        if args.verbose or args.dryrun:
            print("=== APPLICATION ===")
            print(pformat(asdict(app), indent=2, width=80))

            print("=== SCHEDULER REQUEST ===")
            print(runner.dryrun(app, args.scheduler, args.scheduler_args))

        if not args.dryrun:
            app_handle = runner.run(app, args.scheduler, args.scheduler_args)
            print("=== RUN RESULT ===")
            print(f"Launched app: {app_handle}")
            status = runner.status(app_handle)
            print(f"App status: {status}")
            if args.scheduler == "local":
                runner.wait(app_handle)
            else:
                print(f"Job URL: {status.ui_url}")
