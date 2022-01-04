#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import dataclasses
import logging
import pprint
import sys

from torchx.cli.cmd_base import SubCommand
from torchx.runner import get_runner
from torchx.specs import api

logger: logging.Logger = logging.getLogger(__name__)


class CmdDescribe(SubCommand):
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "app_handle",
            type=str,
            help="torchx app handle (e.g. local://session-name/app-id)",
        )

    def run(self, args: argparse.Namespace) -> None:
        app_handle = args.app_handle
        scheduler, session_name, app_id = api.parse_app_handle(app_handle)
        runner = get_runner(name=session_name)
        app = runner.describe(app_handle)

        if app:
            pprint.pprint(dataclasses.asdict(app), indent=2, width=80)
        else:
            logger.error(
                f"AppDef: {app_id} on session: {session_name},"
                f" does not exist or has been removed from {scheduler}'s data plane"
            )
            sys.exit(1)
