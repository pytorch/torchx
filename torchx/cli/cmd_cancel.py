#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging

from torchx.cli.argparse_util import scheduler_params
from torchx.cli.cmd_base import SubCommand
from torchx.runner import get_runner
from torchx.specs.api import parse_app_handle

logger: logging.Logger = logging.getLogger(__name__)


class CmdCancel(SubCommand):
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "app_handle",
            type=str,
            help="torchx app handle (e.g. local://session-name/app-id)",
        )

    def run(self, args: argparse.Namespace) -> None:
        app_handle = args.app_handle
        scheduler, _, app_id = parse_app_handle(app_handle)
        params = scheduler_params(scheduler)
        runner = get_runner(name=None, component_defaults=None, **params)
        runner.cancel(app_handle)
