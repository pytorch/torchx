# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import logging
import os
import sys
import threading
from collections import Counter
from dataclasses import asdict
from itertools import groupby
from pathlib import Path
from pprint import pformat
from typing import Dict, List, Optional, Tuple

import torchx.specs as specs
from torchx.cli.argparse_util import ArgOnceAction, torchxconfig_run
from torchx.cli.cmd_base import SubCommand
from torchx.cli.cmd_log import get_logs
from torchx.runner import config, get_runner, Runner
from torchx.runner.config import load_sections
from torchx.schedulers import get_default_scheduler_name, get_scheduler_factories
from torchx.specs.finder import (
    _Component,
    ComponentNotFoundException,
    ComponentValidationException,
    get_builtin_source,
    get_components,
)
from torchx.util.log_tee_helpers import tee_logs
from torchx.util.types import none_throws


MISSING_COMPONENT_ERROR_MSG = (
    "missing component name, either provide it from the CLI or in .torchxconfig"
)


logger: logging.Logger = logging.getLogger(__name__)


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

    if len(args) > 0:  # check len again in case we removed the leading "--" above
        if args[0].startswith("-"):
            component_args = args
        else:  # first element is NOT an option; then it must be a component name
            component = args[0]
            component_args = args[1:]

    # Error if there are repeated command line arguments each group of arguments,
    # where the groups are separated by "--"
    arg_groups = [list(g) for _, g in groupby(component_args, key=lambda x: x == "--")]
    for arg_group in arg_groups:
        all_options = [
            x
            for x in arg_group
            if x.startswith("-") and x.strip() != "-" and x.strip() != "--"
        ]
        arg_count = Counter(all_options)
        duplicates = [arg for arg, count in arg_count.items() if count > 1]
        if len(duplicates) > 0:
            subparser.error(f"Repeated Command Line Arguments: {duplicates}")

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
            default=get_default_scheduler_name(),
            choices=list(scheduler_names),
            action=torchxconfig_run,
            help="Name of the scheduler to use.",
        )
        subparser.add_argument(
            "-cfg",
            "--scheduler_args",
            type=str,
            action=ArgOnceAction,
            help="Arguments to pass to the scheduler (Ex:`cluster=foo,user=bar`)."
            " For a list of scheduler run options run: `torchx runopts`",
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
            "--workspace",
            "--buck-target",
            default=f"file://{Path.cwd()}",
            action=torchxconfig_run,
            help="local workspace to build/patch (buck-target of main binary if using buck)",
        )
        subparser.add_argument(
            "--parent_run_id",
            type=str,
            action=ArgOnceAction,
            help="optional parent run ID that this run belongs to."
            " It can be used to group runs for experiment tracking purposes",
        )
        subparser.add_argument(
            "--tee_logs",
            action="store_true",
            default=False,
            help="Add additional prefix to log lines to indicate which replica is printing the log",
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

        scheduler_opts = runner.scheduler_run_opts(args.scheduler)
        cfg = scheduler_opts.cfg_from_str(args.scheduler_args)
        config.apply(scheduler=args.scheduler, cfg=cfg)

        component, component_args = _parse_component_name_and_args(
            args.component_name_and_args,
            none_throws(self._subparser),
        )
        try:
            if args.dryrun:
                dryrun_info = runner.dryrun_component(
                    component,
                    component_args,
                    args.scheduler,
                    workspace=args.workspace,
                    cfg=cfg,
                    parent_run_id=args.parent_run_id,
                )
                print(
                    "\n=== APPLICATION ===\n"
                    f"{pformat(asdict(dryrun_info._app), indent=2, width=80)}"
                )

                print("\n=== SCHEDULER REQUEST ===\n" f"{dryrun_info}")
            else:
                app_handle = runner.run_component(
                    component,
                    component_args,
                    args.scheduler,
                    workspace=args.workspace,
                    cfg=cfg,
                    parent_run_id=args.parent_run_id,
                )
                # DO NOT delete this line. It is used by slurm tests to retrieve the app id
                print(app_handle)

                if args.scheduler.startswith("local"):
                    self._wait_and_exit(
                        runner, app_handle, log=True, tee_logs=args.tee_logs
                    )
                else:
                    logger.info(f"Launched app: {app_handle}")
                    app_status = runner.status(app_handle)
                    if app_status:
                        logger.info(app_status.format())
                    if args.wait or args.log:
                        self._wait_and_exit(
                            runner, app_handle, log=args.log, tee_logs=args.tee_logs
                        )

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

        with get_runner(component_defaults=component_defaults) as runner:
            self._run(runner, args)

    def _wait_and_exit(
        self, runner: Runner, app_handle: str, log: bool, tee_logs: bool = False
    ) -> None:
        logger.info("Waiting for the app to finish...")

        log_thread = (
            self._start_log_thread(runner, app_handle, tee_logs_enabled=tee_logs)
            if log
            else None
        )

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

    def _start_log_thread(
        self, runner: Runner, app_handle: str, tee_logs_enabled: bool = False
    ) -> threading.Thread:
        if tee_logs_enabled:
            thread = tee_logs(
                dst=sys.stderr,
                app_handle=app_handle,
                regex=None,
                runner=runner,
                should_tail=True,
                streams=None,
                colorize=not sys.stderr.closed and sys.stderr.isatty(),
            )
        else:
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
