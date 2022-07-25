#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging

from torchx.cli.cmd_base import SubCommand
from torchx.runner import get_runner
from torchx.schedulers import get_default_scheduler_name, get_scheduler_factories

logger: logging.Logger = logging.getLogger(__name__)


class CmdList(SubCommand):
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        scheduler_names = get_scheduler_factories().keys()
        subparser.add_argument(
            "-s",
            "--scheduler",
            type=str,
            default=get_default_scheduler_name(),
            choices=list(scheduler_names),
            help=f"Name of the scheduler to use. One of: [{','.join(scheduler_names)}]."
            " For listing app handles for ray scheduler, RAY_ADDRESS env variable should be set.",
        )

    def run(self, args: argparse.Namespace) -> None:
        with get_runner() as runner:
            jobs = runner.list(args.scheduler)
            print(*jobs, sep="\n")
