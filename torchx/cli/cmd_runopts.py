#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging

from torchx.cli.cmd_base import SubCommand
from torchx.runner.api import get_runner

logger: logging.Logger = logging.getLogger(__name__)


class CmdRunopts(SubCommand):
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "scheduler",
            type=str,
            nargs="?",
            help="scheduler to dump the runopts for, dumps for all schedulers if not specified",
        )

    def run(self, args: argparse.Namespace) -> None:
        scheduler = args.scheduler
        run_opts = get_runner().run_opts()

        if not scheduler:
            for scheduler, opts in run_opts.items():
                logger.info(f"{scheduler}:\n{repr(opts)}")
        else:
            logger.info(repr(run_opts[scheduler]))
