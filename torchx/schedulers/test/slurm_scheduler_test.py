# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import unittest
from unittest.mock import MagicMock, call, patch

from torchx import specs
from torchx.schedulers.api import DescribeAppResponse
from torchx.schedulers.slurm_scheduler import (
    SlurmBatchRequest,
    SlurmReplicaRequest,
    SlurmScheduler,
    create_scheduler,
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
            env={
                "FOO": "bar",
            },
            num_replicas=5,
            resource=specs.Resource(
                cpu=2,
                memMB=10,
                gpu=3,
            ),
        )
        sbatch, srun = SlurmReplicaRequest.from_role(
            "role-name", role, cfg={}
        ).materialize()
        self.assertEqual(
            sbatch,
            [
                "--job-name=role-name",
                "--ntasks-per-node=1",
                "--cpus-per-task=2",
                "--mem=10",
                "--gpus-per-task=3",
            ],
        )
        self.assertEqual(
            srun,
            ["--chdir=/some/path", "--export=FOO=bar", "echo", "'hello slurm'", "test"],
        )

    def test_replica_request_app_id(self) -> None:
        role = specs.Role(
            name="foo",
            image="/some/path",
            entrypoint="echo",
            args=[f"hello {specs.macros.app_id}"],
        )
        _, srun = SlurmReplicaRequest.from_role("role-name", role, cfg={}).materialize()
        self.assertIn(
            "echo 'hello '\"$SLURM_JOB_ID\"''",
            " ".join(srun),
        )

    def test_replica_request_run_config(self) -> None:
        scheduler = create_scheduler("foo")
        role = specs.Role(
            name="foo",
            image="/some/path",
            entrypoint="echo",
            args=["hello"],
        )
        cfg = {
            "partition": "bubblegum",
            "time": "5:13",
        }

        sbatch, _ = SlurmReplicaRequest.from_role("role-name", role, cfg).materialize()

        run_opts = scheduler.run_opts()

        for k, v in cfg.items():
            self.assertIsNotNone(run_opts.get(k))
            self.assertIn(
                f"--{k}={v}",
                sbatch,
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
        info = scheduler.submit_dryrun(app, cfg={})
        req = info.request
        self.assertIsInstance(req, SlurmBatchRequest)
        self.assertEqual(req.cmd, ["sbatch", "--parsable"])
        self.assertEqual(
            set(req.replicas.keys()),
            {"foo-a-0", "foo-a-1", "foo-b-0"},
        )

        script = req.materialize()
        self.assertEqual(
            script,
            """#!/bin/bash
#SBATCH --job-name=foo-a-0 --ntasks-per-node=1
#SBATCH hetjob
#SBATCH --job-name=foo-a-1 --ntasks-per-node=1
#SBATCH hetjob
#SBATCH --job-name=foo-b-0 --ntasks-per-node=1

# exit on error
set -e

srun --chdir=/some/path echo 0 'hello '"$SLURM_JOB_ID"'' :\\
     --chdir=/some/path echo 1 'hello '"$SLURM_JOB_ID"'' :\\
     --chdir=/some/path echo
""",
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
        app_id = scheduler.submit(app, cfg={})
        self.assertEqual(app_id, "1234")

        self.assertEqual(run.call_count, 1)
        args, kwargs = run.call_args
        self.assertEqual(kwargs, {"stdout": subprocess.PIPE, "check": True})
        (args,) = args
        self.assertEqual(
            args[:-1],
            [
                "sbatch",
                "--parsable",
            ],
        )

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
        run.return_value.stdout = b"""
JobID|JobName|Partition|Account|AllocCPUS|State|ExitCode|
176+0|echo-echo-0|compute||1|COMPLETED|0:0|
176+0.batch|batch|||1|COMPLETED|0:0|
176+0.0|echo|||1|COMPLETED|0:0|
176+1|echo-echo-1|compute||1|COMPLETED|0:0|
176+1.0|echo|||1|COMPLETED|0:0|
176+2|echo-echo-2|compute||1|COMPLETED|0:0|
176+2.0|echo|||1|COMPLETED|0:0|
""".strip()

        scheduler = create_scheduler("foo")
        out = scheduler.describe(app_id="176")

        self.assertEqual(run.call_count, 1)
        self.assertEqual(
            run.call_args,
            call(
                ["sacct", "--parsable2", "-j", "176"],
                stdout=subprocess.PIPE,
                check=True,
            ),
        )

        self.assertIsNotNone(out)
        self.assertEqual(out.app_id, "176")
        self.assertEqual(out.msg, "COMPLETED")
        self.assertEqual(out.state, specs.AppState.SUCCEEDED)

    @patch("subprocess.run")
    def test_describe_running(self, run: MagicMock) -> None:
        run.return_value.stdout = b"""JobID|JobName|Partition|Account|AllocCPUS|State|ExitCode
54|echo-echo-0|compute||1|RUNNING|0:0"""

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
