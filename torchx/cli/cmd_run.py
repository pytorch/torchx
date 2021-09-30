# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import sys
from dataclasses import asdict
from pprint import pformat
from typing import Dict, List, cast, Type

import torchx.specs as specs
from pyre_extensions import none_throws
from torchx.cli.cmd_base import SubCommand
from torchx.runner import Runner, get_runner
from torchx.schedulers import get_scheduler_factories, get_default_scheduler_name
from torchx.specs.finder import (
    _Component,
    get_components,
    ComponentValidationException,
    ComponentNotFoundException,
)
from torchx.util.types import to_dict


logger: logging.Logger = logging.getLogger(__name__)


def _convert_to_option_type(
    value: str, option_type: Type[specs.ConfigValue]
) -> specs.ConfigValue:
    if option_type == bool:
        return value.lower() == "true"
    elif option_type == List[str]:
        return value.split(";")
    else:
        # pyre-ignore[19]
        return option_type(value)


def _parse_run_config(arg: str, scheduler_run_opts: specs.runopts) -> specs.RunConfig:
    conf = specs.RunConfig()
    if not arg:
        return conf

    for key, value in to_dict(arg).items():
        option = scheduler_run_opts.get(key)
        if option is None:
            raise ValueError(f"Unknown {key}, run `torchx runopts` for more info")
        option_type = option.opt_type
        typed_value = _convert_to_option_type(value, option_type)
        conf.set(key, typed_value)
    return conf


class CmdBuiltins(SubCommand):
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        pass

    def _builtins(self) -> Dict[str, _Component]:
        return get_components()

    def run(self, args: argparse.Namespace) -> None:
        builtin_components = self._builtins()
        num_builtins = len(builtin_components)
        logger.info(f"Found {num_builtins} builtin configs:")
        for i, component in enumerate(builtin_components.values()):
            logger.info(f" {i + 1:2d}. {component.name}")


class CmdRun(SubCommand):
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        scheduler_names = get_scheduler_factories().keys()
        subparser.add_argument(
            "--scheduler",
            type=str,
            help=f"Name of the scheduler to use. One of: [{','.join(scheduler_names)}]",
            default=get_default_scheduler_name(),
        )
        subparser.add_argument(
            "--scheduler_args",
            type=str,
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

    def _run(self, runner: Runner, args: argparse.Namespace) -> None:
        run_opts = get_runner().run_opts()
        scheduler_opts = run_opts[args.scheduler]
        scheduler_args = _parse_run_config(args.scheduler_args, scheduler_opts)
        try:
            result = runner.run_component(
                args.conf_file,
                args.conf_args,
                args.scheduler,
                scheduler_args,
                dryrun=args.dryrun,
            )
        except (ComponentValidationException, ComponentNotFoundException) as e:
            error_msg = (
                f"\nFailed to run component `{args.conf_file}` got errors: \n {e}"
            )
            print(error_msg)
            return

        if args.dryrun:
            app_dryrun_info = cast(specs.AppDryRunInfo, result)
            logger.info("=== APPLICATION ===")
            logger.info(pformat(asdict(app_dryrun_info._app), indent=2, width=80))

            logger.info("=== SCHEDULER REQUEST ===")
            logger.info(app_dryrun_info)
        else:
            app_handle = cast(specs.AppHandle, result)
            print(app_handle)

            if args.scheduler.startswith("local"):
                self._wait_and_exit(runner, app_handle)
            else:
                logger.info("=== RUN RESULT ===")
                logger.info(f"Launched app: {app_handle}")
                status = runner.status(app_handle)
                logger.info(status)
                logger.info(f"Job URL: {none_throws(status).ui_url}")

                if args.wait:
                    self._wait_and_exit(runner, app_handle)

    def run(self, args: argparse.Namespace) -> None:
        with get_runner() as runner:
            self._run(runner, args)

    def _wait_and_exit(self, runner: Runner, app_handle: str) -> None:
        logger.info("Waiting for the app to finish...")

        status = runner.wait(app_handle, wait_interval=1)
        if not status:
            raise RuntimeError(f"unknown status, wait returned {status}")

        logger.info(f"Job finished: {status.state}")

        if status.state != specs.AppState.SUCCEEDED:
            logger.error(status)
            sys.exit(1)
        else:
            logger.debug(status)
