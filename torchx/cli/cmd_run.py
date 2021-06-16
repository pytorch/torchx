# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import glob
import importlib
import os
from dataclasses import dataclass
from inspect import getmembers, isfunction
from typing import Dict, List, Optional, Union

import torchx.specs as specs
from pyre_extensions import none_throws
from torchx.cli.cmd_base import SubCommand
from torchx.runner import get_runner
from torchx.specs.file_linter import get_fn_docstring, validate
from torchx.util import entrypoints
from torchx.util.io import COMPONENTS_DIR, get_abspath, read_conf_file
from torchx.util.types import to_dict


def parse_args_children(arg: str) -> Dict[str, Union[str, List[str]]]:
    conf = {}
    for key, value in to_dict(arg).items():
        if ";" in value:
            value = value.split(";")
        conf[key] = value
    return conf


def _parse_run_config(arg: str) -> specs.RunConfig:
    conf = specs.RunConfig()
    for key, value in parse_args_children(arg).items():
        conf.set(key, value)
    return conf


def _to_module(filepath: str) -> str:
    path, _ = os.path.splitext(filepath)
    return path.replace(os.path.sep, ".")


def _get_builtin_description(filepath: str, function_name: str) -> Optional[str]:
    source = read_conf_file(filepath)
    if len(validate(source, torchx_function=function_name)) != 0:
        return None

    func_definition, _ = none_throws(get_fn_docstring(source, function_name))
    return func_definition


@dataclass
class BuiltinComponent:
    definition: str
    description: str


def _get_component_definition(module: str, function_name: str) -> str:
    if module.startswith("torchx.components"):
        module = module.split("torchx.components.")[1]
    return f"{module}.{function_name}"


def _to_relative(filepath: str) -> str:
    if os.path.isabs(filepath):
        # make path torchx/components/$suffix out of the abs
        rel_path = filepath.split(str(COMPONENTS_DIR))[1]
        return f"{str(COMPONENTS_DIR)}{rel_path}"
    else:
        return os.path.join(str(COMPONENTS_DIR), filepath)


def _get_components_from_file(filepath: str) -> List[BuiltinComponent]:
    components_path = _to_relative(filepath)
    components_module_path = _to_module(components_path)
    module = importlib.import_module(components_module_path)
    functions = getmembers(module, isfunction)
    buitin_functions = []
    for function_name, _ in functions:
        # Ignore private functions.
        if function_name.startswith("_"):
            continue
        component_desc = _get_builtin_description(filepath, function_name)
        if component_desc:
            definition = _get_component_definition(
                components_module_path, function_name
            )
            builtin_component = BuiltinComponent(
                definition=definition,
                description=component_desc,
            )
            buitin_functions.append(builtin_component)
    return buitin_functions


def _allowed_path(path: str) -> bool:
    filename = os.path.basename(path)
    if filename.startswith("_"):
        return False
    return True


def _builtins() -> List[BuiltinComponent]:
    components_dir = entrypoints.load(
        "torchx.file", "get_dir_path", default=get_abspath
    )(COMPONENTS_DIR)

    builtins: List[BuiltinComponent] = []
    search_pattern = os.path.join(components_dir, "**", "*.py")
    for filepath in glob.glob(search_pattern, recursive=True):
        if not _allowed_path(filepath):
            continue
        components = _get_components_from_file(filepath)
        builtins += components
    return builtins


class CmdBuiltins(SubCommand):
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        pass  # no arguments

    def run(self, args: argparse.Namespace) -> None:
        builtin_configs = _builtins()
        num_builtins = len(builtin_configs)
        print(f"Found {num_builtins} builtin configs:")
        for i, component in enumerate(builtin_configs):
            print(f" {i + 1:2d}. {component.definition} - {component.description}")


class CmdRun(SubCommand):
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--scheduler",
            type=str,
            help="Name of the scheduler to use",
            default="default",
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
        # TODO: T91790598 - remove the if condition when all apps are migrated to pure python
        runner = get_runner()
        app_handle = runner.run_from_path(
            args.conf_file,
            args.conf_args,
            args.scheduler,
            args.scheduler_args,
            dryrun=args.dryrun,
        )

        if not args.dryrun:
            print("=== RUN RESULT ===")
            print(f"Launched app: {app_handle}")
            status = runner.status(app_handle)
            print(f"App status: {status}")
            if args.scheduler == "local":
                runner.wait(app_handle)
            else:
                print(f"Job URL: {none_throws(status).ui_url}")
