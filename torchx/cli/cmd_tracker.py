# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import sys
from typing import Callable, Optional

from tabulate import tabulate

from torchx.cli.cmd_base import SubCommand
from torchx.runner.api import get_configured_trackers
from torchx.tracker.api import build_trackers, TrackerBase
from torchx.util.types import none_throws

logger: logging.Logger = logging.getLogger(__name__)


def _requires_tracker(
    command: Callable[["CmdTracker", argparse.Namespace], None]
) -> Callable[["CmdTracker", argparse.Namespace], None]:
    """Checks that command has valid tracker setup"""

    def wrapper(self: "CmdTracker", args: argparse.Namespace) -> None:
        if not self.tracker:
            logger.error("Exiting since no trackers were configured.")
            sys.exit(1)
        command(self, args)

    return wrapper


class CmdTracker(SubCommand):
    """
    Prototype TorchX tracker subcommand that allows querying data by
    interacting with tracker implementation.

    Important: commands and the arguments may be modified in the future.

    Supported commands:
        - tracker list jobs [–parent-run-id RUN_ID]
        - tracker list metadata RUN_ID
        - tracker list artifacts  [–artifact ARTIFACT_NAME] RUN_ID
    """

    def __init__(self) -> None:
        """
        Queries available tracker implementations and uses the first available one.

        Since the instance needs to be available to setup torchx arguments, subcommands
        utilize `_requires_tracker()` annotation to check that tracker is available
        when invoked.
        """
        self.tracker: Optional[TrackerBase] = None
        configured_trackers = get_configured_trackers()
        if configured_trackers:
            trackers = build_trackers(configured_trackers)
            if trackers:
                self.tracker = next(iter(trackers))
                logger.info(f"Using {self.tracker} to query data")
            else:
                logger.error("No trackers were configured!")

    def add_list_job_arguments(self, subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--parent-run-id", type=str, help="Optional job parent run ID"
        )

    @_requires_tracker
    def list_jobs_command(self, args: argparse.Namespace) -> None:
        parent_run_id = args.parent_run_id
        job_ids = none_throws(self.tracker).run_ids(parent_run_id=parent_run_id)

        tabulated_job_ids = [[job_id] for job_id in job_ids]
        print(tabulate(tabulated_job_ids, headers=["JOB ID"]))

    def add_job_lineage_arguments(self, subparser: argparse.ArgumentParser) -> None:
        group = subparser.add_mutually_exclusive_group()
        group.add_argument(
            "--sources-only", action="store_true", help="Limit to sources"
        )
        group.add_argument(
            "--descendants-only", action="store_true", help="Limit to descendants"
        )

        subparser.add_argument(
            "--artifact", type=str, help="Limit to specific artifact"
        )
        subparser.add_argument("RUN_ID", type=str, help="Job run ID")

    @_requires_tracker
    def job_lineage_command(self, args: argparse.Namespace) -> None:
        raise NotImplementedError("")

    def add_metadata_arguments(self, subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("RUN_ID", type=str, help="Job run ID")

    @_requires_tracker
    def list_metadata_command(self, args: argparse.Namespace) -> None:
        run_id = args.RUN_ID
        metadata = none_throws(self.tracker).metadata(run_id)
        print_data = [[k, v] for k, v in metadata.items()]

        print(tabulate(print_data, headers=["ID", "VALUE"]))

    def add_artifacts_arguments(self, subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument(
            "--artifact", type=str, help="Limit to specific artifact"
        )

        subparser.add_argument("RUN_ID", type=str, help="Job run ID")

    @_requires_tracker
    def list_artifacts_command(self, args: argparse.Namespace) -> None:
        run_id = args.RUN_ID
        artifact_filter = args.artifact

        artifacts = none_throws(self.tracker).artifacts(run_id)
        artifacts = artifacts.values()

        if artifact_filter:
            artifacts = [
                artifact for artifact in artifacts if artifact.name == artifact_filter
            ]
        print_data = [
            [artifact.name, artifact.path, artifact.metadata] for artifact in artifacts
        ]

        print(tabulate(print_data, headers=["ARTIFACT", "PATH", "METADATA"]))

    def add_arguments(self, subparser: argparse.ArgumentParser) -> None:
        tracker_subparsers = subparser.add_subparsers(
            description="Experimental tracker subcommands to query available tracked data.",
        )

        list_subparsers = tracker_subparsers.add_parser(
            "list", help="Use `list --help` to get supported args"
        )
        list_subcmd_subparsers = list_subparsers.add_subparsers(
            description="Tracker list subcommands"
        )
        list_subcommand_mapping = {
            "jobs": (self.list_jobs_command, self.add_list_job_arguments),
            "metadata": (self.list_metadata_command, self.add_metadata_arguments),
            "artifacts": (self.list_artifacts_command, self.add_artifacts_arguments),
        }
        for sub_cmd, [func_handler, args_handler] in list_subcommand_mapping.items():
            sub_cmd_subparser = list_subcmd_subparsers.add_parser(sub_cmd)
            sub_cmd_subparser.set_defaults(func=func_handler)
            args_handler(sub_cmd_subparser)

        job_lineage_subparser = list_subcmd_subparsers.add_parser("lineage")
        job_lineage_subparser.set_defaults(func=self.job_lineage_command)
        self.add_job_lineage_arguments(job_lineage_subparser)

    def run(self, args: argparse.Namespace) -> None:
        # This command defines default func for each tracker subcomand.
        pass
