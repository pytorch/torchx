# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import os
import sys
import threading
from dataclasses import asdict
from pprint import pformat
from typing import Dict, List, Optional, Tuple, Type

import torchx.specs as specs
from pyre_extensions import none_throws
from torchx.cli.cmd_base import SubCommand
from torchx.cli.cmd_log import get_logs
from torchx.runner import Runner, config
from torchx.runner.config import load_sections
from torchx.runner.workspaces import WorkspaceRunner, get_workspace_runner
from torchx.schedulers import get_default_scheduler_name, get_scheduler_factories
from torchx.specs import CfgVal
from torchx.specs.finder import (
    ComponentNotFoundException,
    ComponentValidationException,
    _Component,
    get_builtin_source,
    get_components,
)
from torchx.util.types import to_dict


MISSING_COMPONENT_ERROR_MSG = (
    "missing component name, either provide it from the CLI or in .torchxconfig"
)

logger: logging.Logger = logging.getLogger(__name__)


def _convert_to_option_type(
    value: str, option_type: Type[specs.CfgVal]
) -> specs.CfgVal:
    if option_type == bool:
        return value.lower() == "true"
    elif option_type == List[str]:
        return value.split(";")
    else:
        # pyre-ignore[19]
        return option_type(value)


def _parse_run_config(arg: str, scheduler_opts: specs.runopts) -> Dict[str, CfgVal]:
    conf: Dict[str, CfgVal] = {}
    if not arg:
        return conf

    for key, value in to_dict(arg).items():
        option = scheduler_opts.get(key)
        if option is None:
            raise ValueError(f"Unknown {key}, run `torchx runopts` for more info")
        option_type = option.opt_type
        typed_value = _convert_to_option_type(value, option_type)
        conf[key] = typed_value
    return conf


def _parse_component_name_and_args(
    component_name_and_args: List[str],
    subparser: argparse.ArgumentParser,
    dirs: Optional[List[str]] = None,  # for testing only
) -> Tuple[str, List[str]]:
    """
    Given a list of nargs parsed from commandline, parses out the component name
    and component args. If component name is not found in the list, then
    the default component is loaded from the [cli:run] component section in
    .torchxconfig. If no default config is specified in .torchxconfig, then
    this method errors out to the specified subparser.

    This method deals with the following input list:

    1. [$component_name, *$component_args]
        - Example: ["utils.echo", "--msg", "hello"] or ["utils.echo"]
        - Note: component name and args both in list
    2. [*$component_args]
        - Example: ["--msg", "hello"] or []
        - Note: component name loaded from .torchxconfig, args in list
        - Note: assumes list is only args if the first element
                looks like an option (e.g. starts with "-")

    """
    component = config.get_config(prefix="cli", name="run", key="component", dirs=dirs)
    component_args = []

    # make a copy of the input list to guard against side-effects
    args = list(component_name_and_args)

    if len(args) > 0:
        # `--` is used to delimit between run's options and nargs which includes component args
        # argparse returns the delimiter as part of the nargs so just ignore it if present
        if args[0] == "--":
            args = args[1:]

        if args[0].startswith("-"):
            component_args = args
        else:  # first element is NOT an option; then it must be a component name
            component = args[0]
            component_args = args[1:]

    if not component:
        subparser.error(MISSING_COMPONENT_ERROR_MSG)

    return component, component_args


class CmdBuiltins(SubCommand):
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--print",
            type=str,
            help="prints the builtin's component def to stdout",
        )

    def _builtins(self) -> Dict[str, _Component]:
        return get_components()

    def run(self, args: argparse.Namespace) -> None:
        builtin_name = args.print
        if not builtin_name:
            builtin_components = self._builtins()
            num_builtins = len(builtin_components)
            print(f"Found {num_builtins} builtin components:")
            for i, component in enumerate(builtin_components.values()):
                print(f" {i + 1:2d}. {component.name}")
        else:
            print(get_builtin_source(builtin_name))


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
            "--log",
            action="store_true",
            default=False,
            help="Stream logs while waiting for app to finish.",
        )
        subparser.add_argument(
            "component_name_and_args",
            nargs=argparse.REMAINDER,
        )

    def _run(self, runner: Runner, args: argparse.Namespace) -> None:
        if args.scheduler == "local":
            logger.warning(
                "`local` scheduler is deprecated and will be"
                " removed in the near future,"
                " please use other variants of the local scheduler"
                " (e.g. `local_cwd`)"
            )

        run_opts = runner.run_opts()
        scheduler_opts = run_opts[args.scheduler]
        cfg = _parse_run_config(args.scheduler_args, scheduler_opts)
        config.apply(scheduler=args.scheduler, cfg=cfg)

        config_files = config.find_configs()
        workspace = (
            "file://" + os.path.dirname(config_files[0]) if config_files else None
        )
        component, component_args = _parse_component_name_and_args(
            args.component_name_and_args,
            none_throws(self._subparser),
        )

        try:
            if args.dryrun:
                if isinstance(runner, WorkspaceRunner):
                    dryrun_info = runner.dryrun_component(
                        component,
                        component_args,
                        args.scheduler,
                        workspace=workspace,
                        cfg=cfg,
                    )
                else:
                    dryrun_info = runner.dryrun_component(
                        component, component_args, args.scheduler, cfg=cfg
                    )
                logger.info(
                    "\n=== APPLICATION ===\n"
                    f"{pformat(asdict(dryrun_info._app), indent=2, width=80)}"
                )

                logger.info("\n=== SCHEDULER REQUEST ===\n" f"{dryrun_info}")
            else:
                if isinstance(runner, WorkspaceRunner):
                    app_handle = runner.run_component(
                        component,
                        component_args,
                        args.scheduler,
                        workspace=workspace,
                        cfg=cfg,
                    )
                else:
                    app_handle = runner.run_component(
                        component,
                        component_args,
                        args.scheduler,
                        cfg=cfg,
                    )
                # DO NOT delete this line. It is used by slurm tests to retrieve the app id
                print(app_handle)

                if args.scheduler.startswith("local"):
                    self._wait_and_exit(runner, app_handle, log=True)
                else:
                    logger.info(f"Launched app: {app_handle}")
                    status = runner.status(app_handle)
                    logger.info(status)
                    logger.info(f"Job URL: {none_throws(status).ui_url}")

                    if args.wait:
                        self._wait_and_exit(runner, app_handle, log=args.log)

        except (ComponentValidationException, ComponentNotFoundException) as e:
            error_msg = f"\nFailed to run component `{component}` got errors: \n {e}"
            logger.error(error_msg)
            sys.exit(1)
        except specs.InvalidRunConfigException as e:
            error_msg = (
                f"Scheduler arg is incorrect or missing required option: `{e.cfg_key}`\n"
                f"Run `torchx runopts` to check configuration for `{args.scheduler}` scheduler\n"
                f"Use `-cfg` to specify run cfg as `key1=value1,key2=value2` pair\n"
                "of setup `.torchxconfig` file, see: https://pytorch.org/torchx/main/experimental/runner.config.html"
            )
            logger.error(error_msg)
            sys.exit(1)

    def run(self, args: argparse.Namespace) -> None:
        os.environ["TORCHX_CONTEXT_NAME"] = os.getenv("TORCHX_CONTEXT_NAME", "cli_run")
        component_defaults = load_sections(prefix="component")
        with get_workspace_runner(component_defaults=component_defaults) as runner:
            self._run(runner, args)

    def _wait_and_exit(self, runner: Runner, app_handle: str, log: bool) -> None:
        logger.info("Waiting for the app to finish...")

        log_thread = self._start_log_thread(runner, app_handle) if log else None

        status = runner.wait(app_handle, wait_interval=1)
        if not status:
            raise RuntimeError(f"unknown status, wait returned {status}")

        logger.info(f"Job finished: {status.state}")

        if log_thread:
            log_thread.join()

        if status.state != specs.AppState.SUCCEEDED:
            logger.error(status)
            sys.exit(1)
        else:
            logger.debug(status)

    def _start_log_thread(self, runner: Runner, app_handle: str) -> threading.Thread:
        thread = threading.Thread(
            target=get_logs,
            kwargs={
                "file": sys.stderr,
                "runner": runner,
                "identifier": app_handle,
                "regex": None,
                "should_tail": True,
            },
        )
        thread.daemon = True
        thread.start()
        return thread
