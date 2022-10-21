# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from contextlib import contextmanager
from typing import Generator
from unittest.mock import MagicMock, patch

import torchx
from torchx import specs
from torchx.schedulers.gcp_batch_scheduler import (
    create_scheduler,
    GCPBatchOpts,
    GCPBatchScheduler,
    LABEL_APP_NAME,
    LABEL_VERSION,
)


def _test_app() -> specs.AppDef:
    trainer_role = specs.Role(
        name="trainer",
        image="pytorch/torchx:latest",
        entrypoint="main",
        args=[
            "--output-path",
            specs.macros.img_root,
            "--app-id",
            specs.macros.app_id,
            "--rank0_env",
            specs.macros.rank0_env,
        ],
        env={"FOO": "bar"},
        resource=specs.Resource(
            cpu=2,
            memMB=3000,
            gpu=4,
        ),
        num_replicas=1,
        max_retries=3,
    )
    return specs.AppDef("test", roles=[trainer_role])


@contextmanager
def mock_rand() -> Generator[None, None, None]:
    with patch("torchx.schedulers.gcp_batch_scheduler.make_unique") as make_unique_ctx:
        make_unique_ctx.return_value = "app-name-42"
        yield


class GCPBatchSchedulerTest(unittest.TestCase):
    def test_create_scheduler(self) -> None:
        scheduler = create_scheduler("foo")
        self.assertIsInstance(scheduler, GCPBatchScheduler)

    @mock_rand()
    def test_submit_dryrun(self) -> None:
        from google.cloud import batch_v1

        scheduler = create_scheduler("test")
        app = _test_app()
        proj = "test-proj"
        loc = "us-west-1"
        cfg = GCPBatchOpts(project=proj, location=loc)
        info = scheduler._submit_dryrun(app, cfg)

        req = info.request
        self.assertEqual(req.project, proj)
        self.assertEqual(req.location, loc)

        name = "app-name-42"
        env = {}
        env["TORCHX_ROLE_IDX"] = "0"
        env["TORCHX_ROLE_NAME"] = "trainer"
        env["FOO"] = "bar"
        res = batch_v1.ComputeResource()
        res.cpu_milli = 2000
        res.memory_mib = 3000
        allocationPolicy = batch_v1.AllocationPolicy(
            instances=[
                batch_v1.AllocationPolicy.InstancePolicyOrTemplate(
                    policy=batch_v1.AllocationPolicy.InstancePolicy(
                        machine_type="n1-standard-8",
                        accelerators=[
                            batch_v1.AllocationPolicy.Accelerator(
                                type_="nvidia-tesla-v100",
                                count=4,
                            )
                        ],
                    )
                )
            ],
        )
        runnable = batch_v1.Runnable(
            container=batch_v1.Runnable.Container(
                image_uri="pytorch/torchx:latest",
                commands=[
                    "main",
                    "--output-path",
                    "",
                    "--app-id",
                    "app-name-42",
                    "--rank0_env",
                    "TORCHX_RANK0_HOST",
                ],
            )
        )
        ts = batch_v1.TaskSpec(
            runnables=[runnable],
            environments=env,
            max_retry_count=3,
            compute_resource=res,
        )
        taskGroups = []
        tg = batch_v1.TaskGroup(
            task_spec=ts,
            task_count=1,
            require_hosts_file=True,
        )
        taskGroups.append(tg)
        expected_job_def = batch_v1.Job(
            name=name,
            task_groups=taskGroups,
            allocation_policy=allocationPolicy,
            logs_policy=batch_v1.LogsPolicy(
                destination=batch_v1.LogsPolicy.Destination.CLOUD_LOGGING,
            ),
            labels={
                LABEL_VERSION: torchx.__version__.replace(".", "-"),
                LABEL_APP_NAME: name,
            },
        )

        self.assertEqual(req.job_def, expected_job_def)

    def _mock_scheduler(self) -> GCPBatchScheduler:
        from google.cloud import batch_v1

        scheduler = GCPBatchScheduler("test", client=MagicMock())
        scheduler._client.get_job.return_value = batch_v1.Job(
            name="projects/pytorch-ecosystem-gcp/locations/us-central1/jobs/app-name-42",
            uid="j-82c8495f-8cc9-443c-9e33-4904fcb1test",
            status=batch_v1.JobStatus(
                state=batch_v1.JobStatus.State.SUCCEEDED,
            ),
        )
        return scheduler

    @mock_rand()
    def test_submit(self) -> None:
        scheduler = self._mock_scheduler()
        app = _test_app()
        cfg = GCPBatchOpts(
            project="test-proj",
        )
        info = scheduler._submit_dryrun(app, cfg)
        id = scheduler.schedule(info)
        self.assertEqual(id, "test-proj:us-central1:app-name-42")
        self.assertEqual(scheduler._client.create_job.call_count, 1)

    def test_describe_status(self) -> None:
        scheduler = self._mock_scheduler()
        app_id = "test-proj:us-central1:app-name-42"
        status = scheduler.describe(app_id)
        self.assertIsNotNone(status)
        self.assertEqual(status.state, specs.AppState.SUCCEEDED)
        self.assertEqual(status.app_id, app_id)
