# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import unittest
from unittest.mock import patch, MagicMock, call

from torchx import specs
from torchx.schedulers.api import DescribeAppResponse
from torchx.schedulers.slurm_scheduler import (
    create_scheduler,
    SlurmScheduler,
    SlurmReplicaRequest,
    SlurmBatchRequest,
)


class SlurmSchedulerTest(unittest.TestCase):
    def test_create_scheduler(self) -> None:
        scheduler = create_scheduler("foo")
        self.assertIsInstance(scheduler, SlurmScheduler)

    def test_replica_request(self) -> None:
        role = specs.Role(
            name="foo",
            image="/some/path",
            entrypoint="echo",
            args=["hello slurm", "test"],
            num_replicas=5,
            resource=specs.Resource(
                cpu=2,
                memMB=10,
                gpu=3,
            ),
        )
        script = SlurmReplicaRequest.from_role(role, specs.RunConfig()).materialize()
        self.assertEqual(
            script,
            """#!/bin/sh
#SBATCH --cpus-per-task=2
#SBATCH --mem=10
#SBATCH --gpus-per-task=3

# exit on error
set -e

srun --chdir=/some/path echo 'hello slurm' test
""",
        )

    def test_replica_request_app_id(self) -> None:
        role = specs.Role(
            name="foo",
            image="/some/path",
            entrypoint="echo",
            args=[f"hello {specs.macros.app_id}"],
        )
        script = SlurmReplicaRequest.from_role(role, specs.RunConfig()).materialize()
        self.assertIn(
            "echo 'hello '\"$SLURM_JOB_ID\"''",
            script,
        )

    def test_replica_request_run_config(self) -> None:
        role = specs.Role(
            name="foo",
            image="/some/path",
            entrypoint="echo",
            args=["hello"],
        )
        cfg = specs.RunConfig()
        cfg.set("foo", "bar")
        script = SlurmReplicaRequest.from_role(role, cfg).materialize()
        self.assertIn(
            "#SBATCH --foo=bar",
            script,
        )

    def test_dryrun_multi_role(self) -> None:
        scheduler = create_scheduler("foo")
        app = specs.AppDef(
            name="foo",
            roles=[
                specs.Role(
                    name="a",
                    image="/some/path",
                    entrypoint="echo",
                    args=[specs.macros.replica_id, f"hello {specs.macros.app_id}"],
                    num_replicas=2,
                ),
                specs.Role(
                    name="b",
                    image="/some/path",
                    entrypoint="echo",
                ),
            ],
        )
        info = scheduler.submit_dryrun(app, specs.RunConfig())
        req = info.request
        self.assertIsInstance(req, SlurmBatchRequest)
        self.assertEqual(req.cmd, ["sbatch", "--parsable", "--job-name", "foo"])
        self.assertEqual(
            set(req.replicas.keys()),
            {"role-0-a-0.sh", "role-0-a-1.sh", "role-1-b-0.sh"},
        )

        # check macro substitution
        self.assertIn(
            "echo 1 'hello '\"$SLURM_JOB_ID\"''",
            req.replicas["role-0-a-1.sh"].materialize(),
        )

    @patch("subprocess.run")
    def test_run_multi_role(self, run: MagicMock) -> None:
        run.return_value.stdout = b"1234"
        scheduler = create_scheduler("foo")
        app = specs.AppDef(
            name="foo",
            roles=[
                specs.Role(
                    name="a",
                    image="/some/path",
                    entrypoint="echo",
                    args=[specs.macros.replica_id, f"hello {specs.macros.app_id}"],
                    num_replicas=2,
                ),
                specs.Role(
                    name="b",
                    image="/some/path",
                    entrypoint="echo",
                ),
            ],
        )
        app_id = scheduler.submit(app, specs.RunConfig())
        self.assertEqual(app_id, "1234")

        self.assertEqual(run.call_count, 1)
        args, kwargs = run.call_args
        self.assertEqual(kwargs, {"stdout": subprocess.PIPE, "check": True})
        (args,) = args
        self.assertEqual(len(args), 9)
        self.assertEqual(args[:4], ["sbatch", "--parsable", "--job-name", "foo"])
        self.assertTrue(args[4].endswith("role-0-a-0.sh"))
        self.assertEqual(args[5], ":")
        self.assertTrue(args[6].endswith("role-0-a-1.sh"))
        self.assertEqual(args[7], ":")
        self.assertTrue(args[8].endswith("role-1-b-0.sh"))

    @patch("torchx.schedulers.slurm_scheduler.SlurmScheduler.describe")
    @patch("subprocess.run")
    def test_cancel(self, run: MagicMock, describe: MagicMock) -> None:
        describe.return_value = DescribeAppResponse()

        scheduler = create_scheduler("foo")
        scheduler.cancel("1234")

        self.assertEqual(run.call_count, 1)
        self.assertEqual(run.call_args, call(["scancel", "1234"], check=True))

    @patch("subprocess.run")
    def test_describe_completed(self, run: MagicMock) -> None:
        run.return_value.stdout = b"""JobID|JobName|Partition|Account|AllocCPUS|State|ExitCode
53|echo|compute||1|COMPLETED|0:0
53.batch|batch|||1|COMPLETED|0:0
53.0|echo|||1|COMPLETED|0:0"""

        scheduler = create_scheduler("foo")
        out = scheduler.describe("53")

        self.assertEqual(run.call_count, 1)
        self.assertEqual(
            run.call_args,
            call(
                ["sacct", "--parsable2", "-j", "53"], stdout=subprocess.PIPE, check=True
            ),
        )

        self.assertIsNotNone(out)
        self.assertEqual(out.app_id, "53")
        self.assertEqual(out.msg, "COMPLETED")
        self.assertEqual(out.state, specs.AppState.SUCCEEDED)

    @patch("subprocess.run")
    def test_describe_running(self, run: MagicMock) -> None:
        run.return_value.stdout = b"""JobID|JobName|Partition|Account|AllocCPUS|State|ExitCode
54|echo|compute||1|RUNNING|0:0"""

        scheduler = create_scheduler("foo")
        out = scheduler.describe("54")

        self.assertEqual(run.call_count, 1)
        self.assertEqual(
            run.call_args,
            call(
                ["sacct", "--parsable2", "-j", "54"], stdout=subprocess.PIPE, check=True
            ),
        )

        self.assertIsNotNone(out)
        self.assertEqual(out.app_id, "54")
        self.assertEqual(out.msg, "RUNNING")
        self.assertEqual(out.state, specs.AppState.RUNNING)
