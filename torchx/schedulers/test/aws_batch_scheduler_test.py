# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import threading
import unittest
from contextlib import contextmanager
from typing import Generator
from unittest.mock import MagicMock, patch

import torchx
from torchx import specs
from torchx.schedulers.aws_batch_scheduler import (
    create_scheduler,
    AWSBatchScheduler,
    _role_to_node_properties,
    _local_session,
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
        port_map={"foo": 1234},
        num_replicas=2,
        max_retries=3,
        mounts=[
            specs.BindMount(src_path="/src", dst_path="/dst", read_only=True),
        ],
    )

    return specs.AppDef("test", roles=[trainer_role])


@contextmanager
def mock_rand() -> Generator[None, None, None]:
    with patch("torchx.schedulers.aws_batch_scheduler.make_unique") as make_unique_ctx:
        make_unique_ctx.return_value = "app-name-42"
        yield


class AWSBatchSchedulerTest(unittest.TestCase):
    def test_create_scheduler(self) -> None:
        scheduler = create_scheduler("foo")
        self.assertIsInstance(scheduler, AWSBatchScheduler)

    def test_validate(self) -> None:
        scheduler = create_scheduler("test")
        app = _test_app()
        scheduler._validate(app, "kubernetes")

    @mock_rand()
    def test_submit_dryrun(self) -> None:
        scheduler = create_scheduler("test")
        app = _test_app()
        cfg = {"queue": "testqueue"}
        info = scheduler._submit_dryrun(app, cfg)

        req = info.request
        self.assertEqual(req.queue, "testqueue")
        job_def = req.job_def

        print(job_def)

        self.assertEqual(
            job_def,
            {
                "jobDefinitionName": "app-name-42",
                "type": "multinode",
                "nodeProperties": {
                    "numNodes": 2,
                    "mainNode": 0,
                    "nodeRangeProperties": [
                        {
                            "targetNodes": "0",
                            "container": {
                                "command": [
                                    "main",
                                    "--output-path",
                                    "",
                                    "--app-id",
                                    "app-name-42",
                                    "--rank0_env",
                                    "TORCHX_RANK0_HOST",
                                ],
                                "image": "pytorch/torchx:latest",
                                "environment": [
                                    {"name": "FOO", "value": "bar"},
                                    {"name": "TORCHX_ROLE_IDX", "value": "0"},
                                    {"name": "TORCHX_ROLE_NAME", "value": "trainer"},
                                    {"name": "TORCHX_REPLICA_IDX", "value": "0"},
                                    {"name": "TORCHX_RANK0_HOST", "value": "localhost"},
                                ],
                                "resourceRequirements": [
                                    {"type": "VCPU", "value": "2"},
                                    {"type": "MEMORY", "value": "3000"},
                                    {"type": "GPU", "value": "4"},
                                ],
                                "linuxParameters": {
                                    "sharedMemorySize": 3000,
                                },
                                "logConfiguration": {"logDriver": "awslogs"},
                                "mountPoints": [
                                    {
                                        "containerPath": "/dst",
                                        "readOnly": True,
                                        "sourceVolume": "mount_0",
                                    }
                                ],
                                "volumes": [
                                    {
                                        "name": "mount_0",
                                        "host": {
                                            "sourcePath": "/src",
                                        },
                                    }
                                ],
                            },
                        },
                        {
                            "targetNodes": "1",
                            "container": {
                                "command": [
                                    "main",
                                    "--output-path",
                                    "",
                                    "--app-id",
                                    "app-name-42",
                                    "--rank0_env",
                                    "AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS",
                                ],
                                "image": "pytorch/torchx:latest",
                                "environment": [
                                    {"name": "FOO", "value": "bar"},
                                    {"name": "TORCHX_ROLE_IDX", "value": "0"},
                                    {"name": "TORCHX_ROLE_NAME", "value": "trainer"},
                                    {"name": "TORCHX_REPLICA_IDX", "value": "1"},
                                ],
                                "resourceRequirements": [
                                    {"type": "VCPU", "value": "2"},
                                    {"type": "MEMORY", "value": "3000"},
                                    {"type": "GPU", "value": "4"},
                                ],
                                "linuxParameters": {
                                    "sharedMemorySize": 3000,
                                },
                                "logConfiguration": {"logDriver": "awslogs"},
                                "mountPoints": [
                                    {
                                        "containerPath": "/dst",
                                        "readOnly": True,
                                        "sourceVolume": "mount_0",
                                    }
                                ],
                                "volumes": [
                                    {
                                        "name": "mount_0",
                                        "host": {
                                            "sourcePath": "/src",
                                        },
                                    }
                                ],
                            },
                        },
                    ],
                },
                "retryStrategy": {
                    "attempts": 3,
                    "evaluateOnExit": [{"onExitCode": "0", "action": "EXIT"}],
                },
                "tags": {
                    "torchx.pytorch.org/version": torchx.__version__,
                    "torchx.pytorch.org/app-name": "test",
                },
            },
        )

    def test_volume_mounts(self) -> None:
        role = specs.Role(
            name="foo",
            image="",
            mounts=[
                specs.VolumeMount(src="efsid", dst_path="/dst", read_only=True),
            ],
            resource=specs.Resource(
                cpu=1,
                memMB=1000,
                gpu=0,
            ),
        )
        props = _role_to_node_properties(0, role)
        self.assertEqual(
            # pyre-fixme[16]: `object` has no attribute `__getitem__`.
            props["container"]["volumes"],
            [
                {
                    "name": "mount_0",
                    "efsVolumeConfiguration": {
                        "fileSystemId": "efsid",
                    },
                }
            ],
        )
        self.assertEqual(
            props["container"]["mountPoints"],
            [
                {
                    "containerPath": "/dst",
                    "readOnly": True,
                    "sourceVolume": "mount_0",
                }
            ],
        )

    def _mock_scheduler(self) -> AWSBatchScheduler:
        scheduler = AWSBatchScheduler(
            "test",
            client=MagicMock(),
            log_client=MagicMock(),
        )
        scheduler._client.list_jobs.return_value = {
            "jobSummaryList": [
                {
                    "jobArn": "arn:aws:batch:us-west-2:495572122715:job/6afc27d7-3559-43ca-89fd-1007b6bf2546",
                    "jobId": "6afc27d7-3559-43ca-89fd-1007b6bf2546",
                    "jobName": "echo-v1r560pmwn5t3c",
                    "createdAt": 1643949940162,
                    "status": "SUCCEEDED",
                    "stoppedAt": 1643950324125,
                    "container": {"exitCode": 0},
                    "nodeProperties": {"numNodes": 2},
                    "jobDefinition": "arn:aws:batch:us-west-2:495572122715:job-definition/echo-v1r560pmwn5t3c:1",
                }
            ]
        }
        scheduler._client.describe_jobs.return_value = {
            "jobs": [
                {
                    "jobArn": "thejobarn",
                    "jobName": "app-name-42",
                    "jobId": "6afc27d7-3559-43ca-89fd-1007b6bf2546",
                    "jobQueue": "testqueue",
                    "status": "SUCCEEDED",
                    "attempts": [
                        {
                            "container": {
                                "exitCode": 0,
                                "logStreamName": "log_stream",
                                "networkInterfaces": [],
                            },
                            "startedAt": 1643950310819,
                            "stoppedAt": 1643950324125,
                            "statusReason": "Essential container in task exited",
                        }
                    ],
                    "statusReason": "Essential container in task exited",
                    "createdAt": 1643949940162,
                    "retryStrategy": {
                        "attempts": 1,
                        "evaluateOnExit": [{"onExitCode": "0", "action": "exit"}],
                    },
                    "startedAt": 1643950310819,
                    "stoppedAt": 1643950324125,
                    "dependsOn": [],
                    "jobDefinition": "job-def",
                    "parameters": {},
                    "nodeProperties": {
                        "numNodes": 2,
                        "mainNode": 0,
                        "nodeRangeProperties": [
                            {
                                "targetNodes": "0",
                                "container": {
                                    "image": "ghcr.io/pytorch/torchx:0.1.2dev0",
                                    "command": ["echo", "your name"],
                                    "volumes": [],
                                    "environment": [
                                        {"name": "TORCHX_ROLE_IDX", "value": "0"},
                                        {"name": "TORCHX_REPLICA_IDX", "value": "0"},
                                        {"name": "TORCHX_ROLE_NAME", "value": "echo"},
                                        {
                                            "name": "TORCHX_RANK0_HOST",
                                            "value": "localhost",
                                        },
                                    ],
                                    "mountPoints": [],
                                    "ulimits": [],
                                    "resourceRequirements": [
                                        {"value": "1", "type": "VCPU"},
                                        {"value": "1000", "type": "MEMORY"},
                                    ],
                                    "logConfiguration": {
                                        "logDriver": "awslogs",
                                        "options": {},
                                        "secretOptions": [],
                                    },
                                    "secrets": [],
                                },
                            },
                            {
                                "targetNodes": "1",
                                "container": {
                                    "image": "ghcr.io/pytorch/torchx:0.1.2dev0",
                                    "command": ["echo", "your name"],
                                    "volumes": [],
                                    "environment": [
                                        {"name": "TORCHX_ROLE_IDX", "value": "1"},
                                        {"name": "TORCHX_REPLICA_IDX", "value": "0"},
                                        {"name": "TORCHX_ROLE_NAME", "value": "echo2"},
                                    ],
                                    "mountPoints": [],
                                    "ulimits": [],
                                    "resourceRequirements": [
                                        {"value": "1", "type": "VCPU"},
                                        {"value": "1000", "type": "MEMORY"},
                                    ],
                                    "logConfiguration": {
                                        "logDriver": "awslogs",
                                        "options": {},
                                        "secretOptions": [],
                                    },
                                    "secrets": [],
                                },
                            },
                        ],
                    },
                    "tags": {
                        "torchx.pytorch.org/version": "0.1.2dev0",
                        "torchx.pytorch.org/app-name": "echo",
                    },
                    "platformCapabilities": [],
                }
            ]
        }

        scheduler._log_client.get_log_events.return_value = {
            "nextForwardToken": "some_token",
            "events": [
                {"message": "foo"},
                {"message": "foobar"},
                {"message": "bar"},
            ],
        }

        return scheduler

    @mock_rand()
    def test_submit(self) -> None:
        scheduler = self._mock_scheduler()
        app = _test_app()
        cfg = {
            "queue": "testqueue",
        }

        info = scheduler._submit_dryrun(app, cfg)
        id = scheduler.schedule(info)
        self.assertEqual(id, "testqueue:app-name-42")
        self.assertEqual(scheduler._client.register_job_definition.call_count, 1)
        self.assertEqual(scheduler._client.submit_job.call_count, 1)

    def test_describe(self) -> None:
        scheduler = self._mock_scheduler()
        status = scheduler.describe("testqueue:app-name-42")
        self.assertIsNotNone(status)
        self.assertEqual(status.state, specs.AppState.SUCCEEDED)
        self.assertEqual(status.app_id, "testqueue:app-name-42")
        self.assertEqual(
            status.roles[0],
            specs.Role(
                name="echo",
                num_replicas=1,
                image="ghcr.io/pytorch/torchx:0.1.2dev0",
                entrypoint="echo",
                args=["your name"],
                env={
                    "TORCHX_ROLE_IDX": "0",
                    "TORCHX_REPLICA_IDX": "0",
                    "TORCHX_ROLE_NAME": "echo",
                    "TORCHX_RANK0_HOST": "localhost",
                },
            ),
        )

    def test_log_iter(self) -> None:
        scheduler = self._mock_scheduler()
        logs = scheduler.log_iter("testqueue:app-name-42", "echo", k=1, regex="foo.*")
        self.assertEqual(
            list(logs),
            [
                "foo",
                "foobar",
            ],
        )

    def test_local_session(self) -> None:
        a: object = _local_session()
        self.assertIs(a, _local_session())

        def worker() -> None:
            b = _local_session()
            self.assertIs(b, _local_session())
            self.assertIsNot(a, b)

        t = threading.Thread(target=worker)
        t.start()
        t.join()
