# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from dataclasses import asdict
from pprint import pformat
from typing import Dict, List, Union, cast

import torchx.specs as specs
from pyre_extensions import none_throws
from torchx.cli.cmd_base import SubCommand
from torchx.runner import get_runner
from torchx.specs.finder import get_components, _Component
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


class CmdBuiltins(SubCommand):
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        pass

    def _builtins(self) -> Dict[str, _Component]:
        return get_components()

    def run(self, args: argparse.Namespace) -> None:
        builtin_components = self._builtins()
        num_builtins = len(builtin_components)
        print(f"Found {num_builtins} builtin configs:")
        for i, component in enumerate(builtin_components.values()):
            print(f" {i + 1:2d}. {component.name}")


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
            "--wait",
            action="store_true",
            default=False,
            help="Wait for the app to finish before exiting.",
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
        result = runner.run_component(
            args.conf_file,
            args.conf_args,
            args.scheduler,
            args.scheduler_args,
            dryrun=args.dryrun,
        )

        if args.dryrun:
            app_dryrun_info = cast(specs.AppDryRunInfo, result)
            print("=== APPLICATION ===")
            print(pformat(asdict(app_dryrun_info._app), indent=2, width=80))

            print("=== SCHEDULER REQUEST ===")
            print(app_dryrun_info)
        else:
            app_handle = cast(specs.AppHandle, result)
            if args.scheduler == "local":
                runner.wait(app_handle)
            else:
                print("=== RUN RESULT ===")
                print(f"Launched app: {app_handle}")
                status = runner.status(app_handle)
                print(f"App status: {status}")
                print(f"Job URL: {none_throws(status).ui_url}")

                if args.wait:
                    print("Waiting for the app to finish...")
                    runner.wait(app_handle)
