# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
from argparse import ArgumentParser
from typing import List

from torchx.cli.cmd_describe import CmdDescribe
from torchx.cli.cmd_log import CmdLog
from torchx.cli.cmd_run import CmdBuiltins, CmdRun
from torchx.cli.cmd_runopts import CmdRunopts
from torchx.cli.cmd_status import CmdStatus

sub_parser_description = """Use the following commands to run operations, e.g.:
torchx run ${JOB_NAME}
"""


def create_parser() -> ArgumentParser:
    """
    Helper function parsing the command line options.
    """

    parser = ArgumentParser(description="torchx CLI")
    subparser = parser.add_subparsers(
        title="sub-commands",
        description=sub_parser_description,
    )

    subcmds = {
        "describe": CmdDescribe(),
        "log": CmdLog(),
        "run": CmdRun(),
        "builtins": CmdBuiltins(),
        "runopts": CmdRunopts(),
        "status": CmdStatus(),
    }

    for subcmd_name, cmd in subcmds.items():
        cmd_parser = subparser.add_parser(subcmd_name)
        cmd.add_arguments(cmd_parser)
        cmd_parser.set_defaults(func=cmd.run)

    return parser


def main(argv: List[str] = sys.argv[1:]) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    parser = create_parser()
    args = parser.parse_args(argv)
    if "func" not in args:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
