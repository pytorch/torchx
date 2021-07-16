#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import re
import sys
from datetime import datetime
from string import Template
from typing import List, Optional, Pattern

from torchx.cli.cmd_base import SubCommand
from torchx.runner import get_runner
from torchx.specs import api
from torchx.specs.api import NONE

logger: logging.Logger = logging.getLogger(__name__)


_APP_STATUS_FORMAT_TEMPLATE = """AppDef:
  State: ${state}
  Num Restarts: ${num_restarts}
Roles:${roles}"""

_ROLE_FORMAT_TEMPLATE = "\n  ${role}:${replicas}"

_REPLICA_FORMAT_TEMPLATE_DETAILED = """\n  ${role}[${replica_id}]:
    state: ${state}
    timestamp: ${timestamp} (exit_code: ${exit_code})
    hostname: ${hostname}
    error_msg: ${error_msg}"""

_LINE_WIDTH = 100

# RPC Error message. Example:
# RuntimeError('On WorkerInfo(id=1, name=trainer:0:0):
# <message with the Traceback>
# ')
_RPC_ERROR_MESSAGE_RE: Pattern[str] = re.compile(
    (r"(?P<exception_type>\w*)\('On WorkerInfo\(.+\):\n" r"(.*\n)*" r"'\)")
)

# Sometimes another exception is nested as a message of the outer exception
# rather than proper exception chaining. Example:
#  c10::Error: CUDA error: an illegal memory access was encountered
# Exception
#   raised from create_event_internal at caffe2/c10/cuda/CUDACachingAllocator.cpp:733
#     (most recent call first):
_EMBEDDED_ERROR_MESSAGE_RE: Pattern[str] = re.compile(r"(?P<msg>.+)\nException.*")


def format_error_message(msg: str, header: str, width: int = 80) -> str:
    assert len(header) < width

    match = re.search(_RPC_ERROR_MESSAGE_RE, msg)
    if match:
        start_pos, end_pos = match.span()
        msg = msg[start_pos:end_pos]

    match = re.search(_EMBEDDED_ERROR_MESSAGE_RE, msg)
    if match:
        msg = match.group("msg")

    length = 0
    lines = []
    for i in range(len(msg) + 1):
        if (i == (len(msg))) or (msg[i] == " " and length >= width):
            lines.append(f"{header}{msg[i - length: i]}")
            header = " " * len(header)
            length = 0
        length += 1
    return "\n".join(lines)


def format_replica_status(replica_status: api.ReplicaStatus) -> str:
    if replica_status.structured_error_msg != NONE:
        error_data = json.loads(replica_status.structured_error_msg)
        error_message = format_error_message(
            msg=error_data["message"]["message"], header="    error_msg: "
        )
        timestamp = int(error_data["message"]["extraInfo"]["timestamp"])
        exitcode = error_data["message"]["errorCode"]
        if not exitcode:
            exitcode = "<N/A>"
        data = f"""{str(replica_status.state)} (exitcode: {exitcode})
    timestamp: {datetime.fromtimestamp(timestamp)}
    hostname: {replica_status.hostname}
{error_message}"""
    else:
        data = f"{str(replica_status.state)}"
        if replica_status.state in [
            api.ReplicaState.CANCELLED,
            api.ReplicaState.FAILED,
        ]:
            data += " (no reply file)"

    # mark index 0 for each role with a "*" for a visual queue on role boundaries
    header = " "
    if replica_status.id == 0:
        header = "*"

    return f"\n {header}{replica_status.role}[{replica_status.id}]:{data}"


def format_role_status(
    role_status: api.RoleStatus,
) -> str:
    replica_data = ""

    for replica in sorted(role_status.replicas, key=lambda r: r.id):
        replica_data += format_replica_status(replica)
    return f"{replica_data}"


def get_roles(
    roles: List[api.RoleStatus], filter_roles: Optional[List[str]] = None
) -> List[api.RoleStatus]:
    if not filter_roles:
        return roles
    return [role_status for role_status in roles if role_status.role in filter_roles]


def format_app_status(
    app_status: api.AppStatus,
    filter_roles: Optional[List[str]] = None,
) -> str:
    roles_data = ""
    roles = get_roles(app_status.roles, filter_roles)
    for role_status in roles:
        roles_data += format_role_status(role_status)
    return Template(_APP_STATUS_FORMAT_TEMPLATE).substitute(
        state=app_status.state,
        num_restarts=app_status.num_restarts,
        roles=roles_data,
    )


def parse_list_arg(arg: str) -> Optional[List[str]]:
    if not arg:
        return None
    return arg.split(",")


class CmdStatus(SubCommand):
    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "app_handle",
            type=str,
            help="torchx app handle (e.g. local://session-name/app-id)",
        )
        subparser.add_argument(
            "--roles", type=str, default="", help="comma separated roles to filter"
        )

    def run(self, args: argparse.Namespace) -> None:
        app_handle = args.app_handle
        scheduler, session_name, app_id = api.parse_app_handle(app_handle)
        runner = get_runner(name=session_name)
        app_status = runner.status(app_handle)
        filter_roles = parse_list_arg(args.roles)
        if app_status:
            logger.info(format_app_status(app_status, filter_roles))
        else:
            logger.error(
                f"AppDef: {app_id} on session: {session_name},"
                f" does not exist or has been removed from {scheduler}'s data plane"
            )
            sys.exit(1)
