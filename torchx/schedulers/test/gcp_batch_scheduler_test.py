# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import torchx
from torchx import specs
from torchx.schedulers.api import ListAppResponse
from torchx.schedulers.gcp_batch_scheduler import (
    create_scheduler,
    GCPBatchOpts,
    GCPBatchScheduler,
    LABEL_APP_NAME,
    LABEL_VERSION,
    LOCATIONS,
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
        client = MagicMock()
        scheduler = create_scheduler("foo", client=client)
        self.assertIsInstance(scheduler, GCPBatchScheduler)
        self.assertEqual(scheduler._client, client)

    @mock_rand()
    def test_submit_dryrun(self) -> None:
        from google.cloud import batch_v1

        scheduler = create_scheduler("test")
        app = _test_app()
        proj = "test-proj"
        loc = "us-west-1"
        cfg = GCPBatchOpts(project=proj, location=loc)
        info = scheduler.submit_dryrun(app, cfg)

        req = info.request
        self.assertEqual(req.project, proj)
        self.assertEqual(req.location, loc)

        name = "app-name-42"
        env = {}
        env["TORCHX_ROLE_IDX"] = "0"
        env["TORCHX_ROLE_NAME"] = "trainer"
        env["FOO"] = "bar"
        res = batch_v1.ComputeResource()
        # pyre-fixme[8]: Attribute has type `Field`; used as `int`.
        res.cpu_milli = 2000
        # pyre-fixme[8]: Attribute has type `Field`; used as `int`.
        res.memory_mib = 3000
        allocationPolicy = batch_v1.AllocationPolicy(
            instances=[
                batch_v1.AllocationPolicy.InstancePolicyOrTemplate(
                    install_gpu_drivers=True,
                    policy=batch_v1.AllocationPolicy.InstancePolicy(
                        machine_type="a2-highgpu-4g",
                    ),
                )
            ],
        )
        preRunnable = batch_v1.Runnable(
            script=batch_v1.Runnable.Script(text="/sbin/iptables -A INPUT -j ACCEPT")
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
                    "BATCH_MAIN_NODE_HOSTNAME",
                ],
                options="--net host",
            )
        )
        ts = batch_v1.TaskSpec(
            runnables=[preRunnable, runnable],
            environment=batch_v1.Environment(variables=env),
            max_retry_count=3,
            compute_resource=res,
        )
        taskGroups = []
        tg = batch_v1.TaskGroup(
            task_spec=ts,
            task_count=1,
            task_count_per_node=1,
            task_environments=[
                batch_v1.Environment(variables={"TORCHX_REPLICA_IDX": "0"})
            ],
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

    def test_submit_dryrun_throws(self) -> None:
        scheduler = create_scheduler("test")
        app = _test_app()
        app.roles[0].resource.gpu = 3
        cfg = GCPBatchOpts(project="test-proj", location="us-west-1")
        with self.assertRaises(ValueError):
            scheduler.submit_dryrun(app, cfg)

    def test_app_id_to_job_full_name(self) -> None:
        scheduler = create_scheduler("test")
        app_id = "testproj:testloc:testjob"
        job_name = scheduler._app_id_to_job_full_name(app_id)
        self.assertEqual(job_name, "projects/testproj/locations/testloc/jobs/testjob")

    def test_app_id_to_job_full_name_throws(self) -> None:
        scheduler = create_scheduler("test")
        app_ids = [
            "testproj:testloc:testjob:testanotherjob",
            "testproj:testloc",
            "testproj:testloc/what",
            "testprojtestloctestjob",
        ]
        for app_id in app_ids:
            with self.assertRaises(ValueError):
                scheduler._app_id_to_job_full_name(app_id)

    @patch("google.cloud.batch_v1.BatchServiceClient")
    def test_get_job(self, mock_client: MagicMock) -> None:
        from google.cloud import batch_v1

        scheduler = create_scheduler("test")
        mock_batch_client = mock_client.return_value
        scheduler._get_job("test-proj:us-central1:app-name-42")
        mock_client.assert_called()
        mock_batch_client.get_job.assert_called_once_with(
            request=batch_v1.GetJobRequest(
                name="projects/test-proj/locations/us-central1/jobs/app-name-42",
            )
        )

    @patch("google.cloud.batch_v1.BatchServiceClient")
    def test_cancel_existing(self, mock_client: MagicMock) -> None:
        from google.cloud import batch_v1

        scheduler = create_scheduler("test")
        mock_batch_client = mock_client.return_value
        scheduler._cancel_existing("test-proj:us-central1:app-name-42")
        mock_client.assert_called()
        mock_batch_client.delete_job.assert_called_once_with(
            request=batch_v1.DeleteJobRequest(
                name="projects/test-proj/locations/us-central1/jobs/app-name-42",
                reason="Killed via TorchX",
            )
        )

    def test_job_full_name_to_app_id(self) -> None:
        scheduler = create_scheduler("test")
        job_name = "projects/testproj/locations/testloc/jobs/testjob"
        app_id = scheduler._job_full_name_to_app_id(job_name)
        self.assertEqual(app_id, "testproj:testloc:testjob")

    def test_job_full_name_to_app_id_throws(self) -> None:
        scheduler = create_scheduler("test")
        job_names = [
            "projects/testproj/locations/testloc/jobs/testjob/anotherjob",
            "projects/testproj/locations/testloc/jobs:testjob",
            "projects/testproj/locations/testloc/jobs",
        ]
        for job_name in job_names:
            with self.assertRaises(ValueError):
                scheduler._job_full_name_to_app_id(job_name)

    @patch("google.cloud.batch_v1.BatchServiceClient")
    @patch("torchx.schedulers.gcp_batch_scheduler.GCPBatchScheduler._get_project")
    def test_list(self, mock_project: MagicMock, mock_client: MagicMock) -> None:
        mock_project.return_value = "test-proj"
        scheduler = create_scheduler("test")
        mock_batch_client = mock_client.return_value

        scheduler.list()
        mock_client.assert_called_once()
        mock_batch_client.list_jobs.assert_called()
        self.assertEqual(mock_batch_client.list_jobs.call_count, len(LOCATIONS))

    @patch("google.cloud.logging.Client")
    def test_batch_log_iter(self, mock_log_client: MagicMock) -> None:
        from google.cloud import logging

        mock_logging_client = mock_log_client.return_value
        mock_logger = mock_logging_client.logger.return_value
        # pyre-ignore [28] : LogEntry type initiated with just payload for testing
        log_lines = [logging.LogEntry(payload="log line")]
        mock_logger.list_entries = MagicMock(return_value=iter(log_lines))

        scheduler = create_scheduler("test")
        filter = "labels.job_uid=j-82c8495f-8cc9-443c-9e33-4904fcb1test"
        lines = scheduler._batch_log_iter(filter=filter)
        for l in lines:
            self.assertEqual(l, "log line")
        mock_log_client.assert_called()
        mock_logging_client.logger.assert_called_once_with("batch_task_logs")
        mock_logger.list_entries.assert_called_once_with(filter_=filter)

    def _mock_scheduler(self) -> GCPBatchScheduler:
        from google.cloud import batch_v1

        scheduler = GCPBatchScheduler("test", client=MagicMock())
        scheduler._client.get_job.return_value = batch_v1.Job(
            name="projects/test-proj/locations/us-central1/jobs/app-name-42",
            uid="j-82c8495f-8cc9-443c-9e33-4904fcb1test",
            allocation_policy=batch_v1.AllocationPolicy(
                instances=[
                    batch_v1.AllocationPolicy.InstancePolicyOrTemplate(
                        install_gpu_drivers=True,
                        policy=batch_v1.AllocationPolicy.InstancePolicy(
                            machine_type="a2-highgpu-4g",
                        ),
                    ),
                ],
            ),
            task_groups=[
                batch_v1.TaskGroup(
                    task_spec=batch_v1.TaskSpec(
                        runnables=[
                            batch_v1.Runnable(
                                script=batch_v1.Runnable.Script(
                                    text="/sbin/iptables -A INPUT -j ACCEPT"
                                )
                            ),
                            batch_v1.Runnable(
                                container=batch_v1.Runnable.Container(
                                    image_uri="ghcr.io/pytorch/torchx:0.3.0dev0",
                                    commands=["python"] + ["-c", 'print("hello ")'],
                                    entrypoint="",
                                    options="--net host",
                                )
                            ),
                        ],
                        compute_resource=batch_v1.ComputeResource(
                            cpu_milli=8000,
                            memory_mib=1024,
                        ),
                        environment=batch_v1.Environment(
                            variables={
                                "TORCHX_ROLE_NAME": "testRole",
                            }
                        ),
                        max_retry_count=2,
                    ),
                    task_count=2,
                )
            ],
            status=batch_v1.JobStatus(
                state=batch_v1.JobStatus.State.SUCCEEDED,
            ),
        )
        scheduler._get_project = MagicMock(return_value="test-proj")
        scheduler._client.list_jobs.side_effect = self._list_side_effect
        log_lines = ["log line foo", "log line bar"]
        scheduler._batch_log_iter = MagicMock(return_value=iter(log_lines))
        return scheduler

    # pyre-ignore [11] : ListJobsPager type undefined as batch_v1 is not imported at top level
    def _list_side_effect(self, *args: Any, **kwargs: Any) -> "ListJobsPager":
        from google.api_core import datetime_helpers
        from google.cloud import batch_v1

        listJobsResp = batch_v1.ListJobsResponse(jobs=[])
        if kwargs["parent"] == "projects/test-proj/locations/us-central1":
            listJobsResp = batch_v1.ListJobsResponse(
                jobs=[
                    batch_v1.Job(
                        name="projects/test-proj/locations/us-central1/jobs/app-name-42",
                        uid="j-82c8495f-8cc9-443c-9e33-0test",
                        status=batch_v1.JobStatus(
                            state=batch_v1.JobStatus.State.SUCCEEDED,
                        ),
                        create_time=datetime_helpers.DatetimeWithNanoseconds(
                            2022, 6, 22, 17, 1, 30, 12345
                        ),
                    ),
                    batch_v1.Job(
                        name="projects/test-proj/locations/us-central1/jobs/torchx-utils-abcd",
                        uid="j-82c8495f-8cc9-443c-9e33-1test",
                        status=batch_v1.JobStatus(
                            state=batch_v1.JobStatus.State.FAILED,
                        ),
                        create_time=datetime_helpers.DatetimeWithNanoseconds(
                            2022, 7, 22, 17, 1, 30, 12345
                        ),
                    ),
                ]
            )
        elif kwargs["parent"] == "projects/test-proj/locations/us-west1":
            listJobsResp = batch_v1.ListJobsResponse(
                jobs=[
                    batch_v1.Job(
                        name="projects/test-proj/locations/us-west1/jobs/torchx-utils-abcd",
                        uid="j-82c8495f-8cc9-443c-9e33-3test",
                        status=batch_v1.JobStatus(
                            state=batch_v1.JobStatus.State.RUNNING,
                        ),
                        create_time=datetime_helpers.DatetimeWithNanoseconds(
                            2022, 6, 30, 17, 1, 30, 12345
                        ),
                    ),
                ]
            )
        return batch_v1.services.batch_service.pagers.ListJobsPager(
            # pyre-ignore [6] : ok for method to be None for testing
            method=None,
            response=listJobsResp,
            # pyre-ignore [6] : ok for request to be None for testing
            request=None,
        )

    @mock_rand()
    def test_submit(self) -> None:
        scheduler = self._mock_scheduler()
        app = _test_app()
        cfg = GCPBatchOpts(
            project="test-proj",
        )
        # pyre-fixme: GCPBatchOpts type passed to resolve
        resolved_cfg = scheduler.run_opts().resolve(cfg)
        # pyre-fixme: _submit_dryrun expects GCPBatchOpts
        info = scheduler.submit_dryrun(app, resolved_cfg)
        id = scheduler.schedule(info)
        self.assertEqual(id, "test-proj:us-central1:app-name-42")
        self.assertEqual(scheduler._client.create_job.call_count, 1)

    def test_describe(self) -> None:
        scheduler = self._mock_scheduler()
        app_id = "test-proj:us-central1:app-name-42"
        desc = scheduler.describe(app_id)
        self.assertIsNotNone(desc)
        self.assertEqual(desc.state, specs.AppState.SUCCEEDED)
        self.assertEqual(desc.app_id, app_id)
        self.assertEqual(
            desc.roles[0],
            specs.Role(
                name="testRole",
                num_replicas=2,
                image="ghcr.io/pytorch/torchx:0.3.0dev0",
                entrypoint="python",
                args=["-c", 'print("hello ")'],
                resource=specs.Resource(
                    cpu=8,
                    memMB=1024,
                    gpu=4,
                ),
                env={
                    "TORCHX_ROLE_NAME": "testRole",
                },
                max_retries=2,
            ),
        )

    def test_list_values(self) -> None:
        scheduler = self._mock_scheduler()
        apps = scheduler.list()
        self.assertEqual(
            apps,
            [
                ListAppResponse(
                    app_id="test-proj:us-central1:torchx-utils-abcd",
                    state=specs.AppState.FAILED,
                ),
                ListAppResponse(
                    app_id="test-proj:us-west1:torchx-utils-abcd",
                    state=specs.AppState.RUNNING,
                ),
                ListAppResponse(
                    app_id="test-proj:us-central1:app-name-42",
                    state=specs.AppState.SUCCEEDED,
                ),
            ],
        )

    def test_log_iter_calls(self) -> None:
        scheduler = self._mock_scheduler()
        logs = scheduler.log_iter(
            app_id="test-proj:us-west1:torchx-utils-abcd", k=1, regex="foo.*"
        )
        scheduler._batch_log_iter.assert_called_once_with(
            f'labels.job_uid=j-82c8495f-8cc9-443c-9e33-4904fcb1test \
AND resource.labels.task_id:task/j-82c8495f-8cc9-443c-9e33-4904fcb1test-group0-1 \
AND timestamp>="{str(datetime.fromtimestamp(0).isoformat())}" \
AND textPayload =~ "foo.*"'
        )

    def test_log_iter_values(self) -> None:
        scheduler = self._mock_scheduler()
        app_id = "test-proj:us-central1:app-name-42"
        logs = scheduler.log_iter(app_id)
        scheduler._batch_log_iter.assert_called_once_with(
            f'labels.job_uid=j-82c8495f-8cc9-443c-9e33-4904fcb1test \
AND resource.labels.task_id:task/j-82c8495f-8cc9-443c-9e33-4904fcb1test-group0-0 \
AND timestamp>="{str(datetime.fromtimestamp(0).isoformat())}"'
        )
        self.assertEqual(
            list(logs),
            [
                "log line foo",
                "log line bar",
            ],
        )
