# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import os
import subprocess
import tempfile
import unittest
from contextlib import contextmanager
from typing import Generator
from unittest.mock import MagicMock, call, patch

from torchx import specs
from torchx.schedulers.api import DescribeAppResponse, Stream
from torchx.schedulers.slurm_scheduler import (
    SlurmBatchRequest,
    SlurmReplicaRequest,
    SlurmScheduler,
    create_scheduler,
)


@contextmanager
def tmp_cwd() -> Generator[None, None, None]:
    with tempfile.TemporaryDirectory() as path:
        cwd = os.getcwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(cwd)


def simple_role() -> specs.Role:
    return specs.Role(
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


def simple_app() -> specs.AppDef:
    return specs.AppDef(
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


class SlurmSchedulerTest(unittest.TestCase):
    def test_create_scheduler(self) -> None:
        scheduler = create_scheduler("foo")
        self.assertIsInstance(scheduler, SlurmScheduler)

    def test_replica_request(self) -> None:
        role = simple_role()
        sbatch, srun = SlurmReplicaRequest.from_role(
            "role-0", role, cfg={}
        ).materialize()
        self.assertEqual(
            sbatch,
            [
                "--job-name=role-0",
                "--ntasks-per-node=1",
                "--cpus-per-task=2",
                "--mem=10",
                "--gpus-per-task=3",
            ],
        )
        self.assertEqual(
            srun,
            [
                '--output=slurm-"$SLURM_JOB_ID"-role-0.out',
                '--error=slurm-"$SLURM_JOB_ID"-role-0.err',
                "--export=ALL,FOO=bar",
                "echo",
                "'hello slurm'",
                "test",
            ],
        )

    def test_replica_request_nomem(self) -> None:
        sbatch, srun = SlurmReplicaRequest.from_role(
            "role-name", simple_role(), cfg={"nomem": True}
        ).materialize()
        self.assertEqual(
            sbatch,
            [
                "--job-name=role-name",
                "--ntasks-per-node=1",
                "--cpus-per-task=2",
                "--gpus-per-task=3",
            ],
        )

    def test_replica_request_constraint(self) -> None:
        sbatch, srun = SlurmReplicaRequest.from_role(
            "role-name", simple_role(), cfg={"constraint": "orange"}
        ).materialize()
        self.assertIn(
            "--constraint=orange",
            sbatch,
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
        app = simple_app()
        info = scheduler.submit_dryrun(app, cfg={})
        req = info.request
        self.assertIsInstance(req, SlurmBatchRequest)
        self.assertEqual(req.cmd, ["sbatch", "--parsable"])
        self.assertEqual(
            set(req.replicas.keys()),
            {"a-0", "a-1", "b-0"},
        )

        script = req.materialize()
        self.assertEqual(
            script,
            """#!/bin/bash
#SBATCH --job-name=a-0 --ntasks-per-node=1
#SBATCH hetjob
#SBATCH --job-name=a-1 --ntasks-per-node=1
#SBATCH hetjob
#SBATCH --job-name=b-0 --ntasks-per-node=1

# exit on error
set -e

export PYTHONUNBUFFERED=1
export SLURM_UNBUFFEREDIO=1

srun --output=slurm-"$SLURM_JOB_ID"-a-0.out --error=slurm-"$SLURM_JOB_ID"-a-0.err echo 0 'hello '"$SLURM_JOB_ID"'' :\\
     --output=slurm-"$SLURM_JOB_ID"-a-1.out --error=slurm-"$SLURM_JOB_ID"-a-1.err echo 1 'hello '"$SLURM_JOB_ID"'' :\\
     --output=slurm-"$SLURM_JOB_ID"-b-0.out --error=slurm-"$SLURM_JOB_ID"-b-0.err echo
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
JobID|JobName|Partition|Account|AllocCPUS|State|ExitCode
1853+0|echo-0|compute||1|COMPLETED|0:0
1853+0.batch|batch|||1|COMPLETED|0:0
1853+0.0|echo|||1|COMPLETED|0:0
1853+1|echo-1|compute||1|COMPLETED|0:0
1853+1.0|echo|||1|COMPLETED|0:0
1853+2|echo-2|compute||1|COMPLETED|0:0
1853+2.0|echo|||1|COMPLETED|0:0
""".strip()

        scheduler = create_scheduler("foo")
        out = scheduler.describe(app_id="1853")

        self.assertEqual(run.call_count, 1)
        self.assertEqual(
            run.call_args,
            call(
                ["sacct", "--parsable2", "-j", "1853"],
                stdout=subprocess.PIPE,
                check=True,
            ),
        )

        self.assertIsNotNone(out)
        self.assertEqual(out.app_id, "1853")
        self.assertEqual(out.msg, "COMPLETED")
        self.assertEqual(out.state, specs.AppState.SUCCEEDED)
        self.assertEqual(
            out.roles,
            [
                specs.Role(
                    name="echo",
                    image="",
                    num_replicas=3,
                )
            ],
        )

    @patch("subprocess.run")
    def test_describe_single_replica(self, run: MagicMock) -> None:
        run.return_value.stdout = b"""
JobID|JobName|Partition|Account|AllocCPUS|State|ExitCode
1902|sh-0|compute||1|FAILED|2:0
1902.batch|batch|||1|FAILED|2:0
1902.0|sh|||1|FAILED|2:0
""".strip()

        scheduler = create_scheduler("foo")
        out = scheduler.describe(app_id="1902")

        self.assertEqual(run.call_count, 1)
        self.assertEqual(
            run.call_args,
            call(
                ["sacct", "--parsable2", "-j", "1902"],
                stdout=subprocess.PIPE,
                check=True,
            ),
        )

        self.assertIsNotNone(out)
        self.assertEqual(out.app_id, "1902")
        self.assertEqual(out.msg, "FAILED")
        self.assertEqual(out.state, specs.AppState.FAILED)
        self.assertEqual(
            out.roles,
            [
                specs.Role(
                    name="sh",
                    image="",
                    num_replicas=1,
                )
            ],
        )

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

    @patch("subprocess.run")
    def test_log_iter(self, run: MagicMock) -> None:
        scheduler = create_scheduler("foo")

        with tmp_cwd():
            with open("slurm-54-echo-1.out", "wt") as f:
                f.write("hello\nworld\n")

            logs = list(
                scheduler.log_iter(
                    "54",
                    "echo",
                    1,
                    streams=Stream.STDOUT,
                    since=datetime.datetime.now(),
                )
            )
            self.assertEqual(logs, ["hello", "world"])

            with open("slurm-54-echo-1.err", "wt") as f:
                f.write("foo\nbar\n")

            logs = list(
                scheduler.log_iter(
                    "54",
                    "echo",
                    1,
                    streams=Stream.STDERR,
                )
            )

            self.assertEqual(logs, ["foo", "bar"])

            # no stream specified should default to STDERR
            logs = list(
                scheduler.log_iter(
                    "54",
                    "echo",
                    1,
                )
            )
            self.assertEqual(logs, ["foo", "bar"])

        with self.assertRaises(ValueError):
            scheduler.log_iter("54", "echo", 1, streams=Stream.COMBINED)

    def test_dryrun_comment(self) -> None:
        scheduler = create_scheduler("foo")
        app = simple_app()
        info = scheduler.submit_dryrun(
            app,
            cfg={
                "comment": "banana foo bar",
            },
        )
        self.assertIn(
            "--comment=banana foo bar",
            info.request.cmd,
        )

    def test_dryrun_mail(self) -> None:
        scheduler = create_scheduler("foo")
        app = simple_app()
        info = scheduler.submit_dryrun(
            app,
            cfg={
                "mail-user": "foo@bar.com",
                "mail-type": "END",
            },
        )
        self.assertIn(
            "--mail-user=foo@bar.com",
            info.request.cmd,
        )
        self.assertIn(
            "--mail-type=END",
            info.request.cmd,
        )
