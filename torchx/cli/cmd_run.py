# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import sys
from dataclasses import asdict
from pprint import pformat
from typing import Dict, List, Optional, Type, cast

import torchx.specs as specs
from pyre_extensions import none_throws
from torchx.cli.cmd_base import SubCommand
from torchx.runner import Runner, config, get_runner
from torchx.schedulers import get_default_scheduler_name, get_scheduler_factories
from torchx.specs.finder import (
    ComponentNotFoundException,
    ComponentValidationException,
    _Component,
    get_components,
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


def _parse_run_config(arg: str, scheduler_opts: specs.runopts) -> specs.RunConfig:
    conf = specs.RunConfig()
    if not arg:
        return conf

    for key, value in to_dict(arg).items():
        option = scheduler_opts.get(key)
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
        print(f"Found {num_builtins} builtin configs:")
        for i, component in enumerate(builtin_components.values()):
            print(f" {i + 1:2d}. {component.name}")


class CmdRun(SubCommand):
    def __init__(self) -> None:
        self._subparser: Optional[argparse.ArgumentParser] = None

    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        scheduler_names = get_scheduler_factories().keys()
        self._subparser = subparser
        subparser.add_argument(
            "-s",
            "--scheduler",
            type=str,
            help=f"Name of the scheduler to use. One of: [{','.join(scheduler_names)}]",
            default=get_default_scheduler_name(),
        )
        subparser.add_argument(
            "-cfg",
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
            "conf_args",
            nargs=argparse.REMAINDER,
        )

    def _run(self, runner: Runner, args: argparse.Namespace) -> Optional[str]:
        run_opts = get_runner().run_opts()
        scheduler_opts = run_opts[args.scheduler]
        cfg = _parse_run_config(args.scheduler_args, scheduler_opts)
        config.apply(scheduler=args.scheduler, cfg=cfg)

        if len(args.conf_args) < 1:
            none_throws(self._subparser).error(
                "the following arguments are required: conf_file, conf_args"
            )

        # Python argparse would remove `--` if it was the first argument. This
        # does not work well for torchx, since torchx.specs.api uses another argparser to
        # parse component arguments.
        conf_file, conf_args = args.conf_args[0], args.conf_args[1:]
        try:
            result = runner.run_component(
                conf_file,
                conf_args,
                args.scheduler,
                cfg,
                dryrun=args.dryrun,
            )
        except (ComponentValidationException, ComponentNotFoundException) as e:
            error_msg = f"\nFailed to run component `{conf_file}` got errors: \n {e}"
            print(error_msg)
            return

        if args.dryrun:
            app_dryrun_info = cast(specs.AppDryRunInfo, result)
            logger.info(
                "\n=== APPLICATION ===\n"
                f"{pformat(asdict(app_dryrun_info._app), indent=2, width=80)}"
            )

            logger.info("\n=== SCHEDULER REQUEST ===\n" f"{app_dryrun_info}")
            return
        else:
            app_handle = cast(specs.AppHandle, result)
            # do not delete this line. It is used by slurm tests to retrieve the app id
            print(app_handle)

            if args.scheduler.startswith("local"):
                self._wait_and_exit(runner, app_handle)
            else:
                logger.info(f"Launched app: {app_handle}")
                status = runner.status(app_handle)
                logger.info(status)
                logger.info(f"Job URL: {none_throws(status).ui_url}")

                if args.wait:
                    self._wait_and_exit(runner, app_handle)
            return app_handle

    def run(self, args: argparse.Namespace) -> None:
        os.environ["TORCHX_CONTEXT_NAME"] = os.getenv("TORCHX_CONTEXT_NAME", "cli_run")
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
