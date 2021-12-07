#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import fsspec
from docker.types import DeviceRequest
from torchx import specs
from torchx.schedulers.api import Stream
from torchx.schedulers.docker_scheduler import (
    DockerContainer,
    DockerJob,
    DockerScheduler,
    create_scheduler,
    has_docker,
)
from torchx.schedulers.test.local_scheduler_test import LocalSchedulerTestUtil
from torchx.specs.api import AppDef, AppState, Role


def _test_app() -> specs.AppDef:
    trainer_role = specs.Role(
        name="trainer",
        image="pytorch/torchx:latest",
        entrypoint="main",
        args=["--output-path", specs.macros.img_root, "--app-id", specs.macros.app_id],
        env={"FOO": "bar"},
        resource=specs.Resource(
            cpu=2,
            memMB=3000,
            gpu=4,
        ),
        port_map={"foo": 1234},
        num_replicas=1,
        max_retries=3,
    )

    return specs.AppDef("test", roles=[trainer_role])


class DockerSchedulerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scheduler: DockerScheduler = create_scheduler(
            session_name="test_session",
        )

    def test_submit_dryrun(self) -> None:
        app = _test_app()
        with patch("torchx.schedulers.docker_scheduler.make_unique") as make_unique_ctx:
            make_unique_ctx.return_value = "app_name_42"
            info = self.scheduler._submit_dryrun(app, cfg={})

        want = DockerJob(
            "app_name_42",
            [
                DockerContainer(
                    image="pytorch/torchx:latest",
                    command=[
                        "main",
                        "--output-path",
                        "",
                        "--app-id",
                        "app_name_42",
                    ],
                    kwargs={
                        "device_requests": [
                            DeviceRequest(
                                count=4,
                                capabilities=[["compute"]],
                            )
                        ],
                        "environment": {
                            "FOO": "bar",
                        },
                        "labels": {
                            "torchx.pytorch.org/app-id": "app_name_42",
                            "torchx.pytorch.org/replica-id": "0",
                            "torchx.pytorch.org/role-name": "trainer",
                            "torchx.pytorch.org/version": "0.1.2dev0",
                        },
                        "mem_limit": "3000m",
                        "name": "app_name_42-trainer-0",
                        "hostname": "app_name_42-trainer-0",
                        "nano_cpus": int(2e9),
                        "restart_policy": {
                            "Name": "on-failure",
                            "MaximumRetryCount": 3,
                        },
                        "network": "torchx",
                    },
                )
            ],
        )
        self.assertEqual(str(info), str(want))

    @patch("os.environ", {"FOO_1": "f1", "BAR_1": "b1", "FOOBAR_1": "fb1"})
    def test_copy_env(self) -> None:
        app = _test_app()
        cfg = {"copy_env": ["FOO_*", "BAR_*"]}
        info = self.scheduler._submit_dryrun(app, cfg)
        self.assertEqual(
            info.request.containers[0].kwargs["environment"],
            {
                "FOO": "bar",
                "FOO_1": "f1",
                "BAR_1": "b1",
            },
        )


if has_docker():
    # These are the live tests that require a local docker instance.

    class DockerSchedulerLiveTest(unittest.TestCase, LocalSchedulerTestUtil):
        def setUp(self) -> None:
            self.scheduler: DockerScheduler = create_scheduler(
                session_name="test_session",
            )

        def _docker_app(self, entrypoint: str, *args: str) -> AppDef:
            return AppDef(
                name="test-app",
                roles=[
                    Role(
                        name="image_test_role",
                        image="busybox",
                        entrypoint=entrypoint,
                        args=list(args),
                    ),
                ],
            )

        def test_docker_submit(self) -> None:
            app = self._docker_app("echo", "foo")
            app_id = self.scheduler.submit(app, cfg={})

            desc = self.wait(app_id)
            self.assertIsNotNone(desc)
            self.assertEqual(AppState.SUCCEEDED, desc.state)
            self.assertEqual(len(desc.roles), 1)
            self.assertEqual(len(desc.roles_statuses), 1)
            self.assertEqual(len(desc.roles_statuses[0].replicas), 1)
            self.assertEqual(
                desc.roles_statuses[0].replicas[0].state, AppState.SUCCEEDED
            )

            self.assertEqual(desc.app_id, app_id)

        def test_docker_logs(self) -> None:
            app = self._docker_app("echo", "foo\nbar")
            start = datetime.utcnow()
            app_id = self.scheduler.submit(app, cfg={})
            desc = self.wait(app_id)
            self.assertIsNotNone(desc)
            # docker truncates to the second so pad out 1 extra second
            end = datetime.utcnow() + timedelta(seconds=1)

            self.assertEqual(AppState.SUCCEEDED, desc.state)

            logs = list(
                self.scheduler.log_iter(
                    app_id,
                    "image_test_role",
                    0,
                    since=start,
                    until=end,
                )
            )
            self.assertEqual(
                logs,
                [
                    "foo",
                    "bar",
                ],
            )
            logs = list(
                self.scheduler.log_iter(
                    app_id,
                    "image_test_role",
                    0,
                    regex="bar",
                )
            )
            self.assertEqual(
                logs,
                [
                    "bar",
                ],
            )

            logs = list(
                self.scheduler.log_iter(
                    app_id,
                    "image_test_role",
                    0,
                    since=end,
                )
            )
            self.assertEqual(logs, [])
            logs = list(
                self.scheduler.log_iter(
                    app_id,
                    "image_test_role",
                    0,
                    until=start,
                )
            )
            self.assertEqual(logs, [])
            logs = list(
                self.scheduler.log_iter(
                    app_id,
                    "image_test_role",
                    0,
                    should_tail=True,
                )
            )
            self.assertEqual(
                logs,
                [
                    "foo",
                    "bar",
                ],
            )

        def test_docker_logs_streams(self) -> None:
            app = self._docker_app("sh", "-c", "echo stdout; >&2 echo stderr")

            start = datetime.utcnow()
            app_id = self.scheduler.submit(app, cfg={})
            desc = self.wait(app_id)
            self.assertIsNotNone(desc)

            logs = set(
                self.scheduler.log_iter(app_id, "image_test_role", 0, streams=None)
            )
            self.assertEqual(
                logs,
                {
                    "stdout",
                    "stderr",
                },
            )

            logs = set(
                self.scheduler.log_iter(
                    app_id, "image_test_role", 0, streams=Stream.COMBINED
                )
            )
            self.assertEqual(
                logs,
                {
                    "stdout",
                    "stderr",
                },
            )

            logs = list(
                self.scheduler.log_iter(
                    app_id, "image_test_role", 0, streams=Stream.STDERR
                )
            )
            self.assertEqual(
                logs,
                [
                    "stderr",
                ],
            )

            logs = list(
                self.scheduler.log_iter(
                    app_id, "image_test_role", 0, streams=Stream.STDOUT
                )
            )
            self.assertEqual(
                logs,
                [
                    "stdout",
                ],
            )

        def test_docker_cancel(self) -> None:
            app = self._docker_app("sleep", "10000")
            app_id = self.scheduler.submit(app, cfg={})
            _ = self.scheduler.describe(app_id)

            self.wait(app_id, wait_for=lambda state: state == AppState.RUNNING)
            self.scheduler.cancel(app_id)

            desc = self.wait(app_id)
            self.assertIsNotNone(desc)
            self.assertEqual(desc.state, AppState.FAILED)

        def test_docker_submit_error(self) -> None:
            app = self._docker_app("sh", "-c", "exit 1")
            app_id = self.scheduler.submit(app, cfg={})

            desc = self.wait(app_id)
            self.assertIsNotNone(desc)
            self.assertEqual(AppState.FAILED, desc.state)
            self.assertEqual(len(desc.roles), 1)
            self.assertEqual(len(desc.roles_statuses), 1)
            self.assertEqual(len(desc.roles_statuses[0].replicas), 1)
            self.assertEqual(desc.roles_statuses[0].replicas[0].state, AppState.FAILED)

        def test_docker_submit_error_retries(self) -> None:
            app = self._docker_app("sh", "-c", "exit 1")
            app.roles[0].max_retries = 1
            app_id = self.scheduler.submit(app, cfg={})

            desc = self.wait(app_id)
            self.assertIsNotNone(desc)
            self.assertEqual(AppState.FAILED, desc.state)

        def test_docker_submit_dist(self) -> None:
            app = AppDef(
                name="test-app",
                roles=[
                    Role(
                        name="ping",
                        image="busybox",
                        entrypoint="sh",
                        args=[
                            "-c",
                            f"sleep 1; ping -c1 {specs.macros.app_id}-ping-0; sleep 1",
                        ],
                        num_replicas=2,
                    ),
                ],
            )
            app_id = self.scheduler.submit(app, cfg={})
            print(app_id)

            desc = self.wait(app_id)
            self.assertIsNotNone(desc)
            self.assertEqual(AppState.SUCCEEDED, desc.state)
            self.assertEqual(len(desc.roles), 1)
            self.assertEqual(len(desc.roles_statuses), 1)
            self.assertEqual(len(desc.roles_statuses[0].replicas), 2)
            self.assertEqual(
                desc.roles_statuses[0].replicas[0].state, AppState.SUCCEEDED
            )
            self.assertEqual(
                desc.roles_statuses[0].replicas[1].state, AppState.SUCCEEDED
            )

        def test_docker_workspace(self) -> None:
            fs = fsspec.filesystem("memory")
            fs.mkdirs("test_workspace/bar", exist_ok=True)
            with fs.open("test_workspace/bar/foo.sh", "w") as f:
                f.write("exit 0")

            img = self.scheduler.build_workspace_image(
                "busybox",
                "memory://test_workspace",
            )

            app = AppDef(
                name="test-app",
                roles=[
                    Role(
                        name="ping",
                        image=img,
                        entrypoint="sh",
                        args=[
                            "bar/foo.sh",
                        ],
                    ),
                ],
            )
            app_id = self.scheduler.submit(app, {})
            print(app_id)

            desc = self.wait(app_id)
            self.assertIsNotNone(desc)
            self.assertEqual(AppState.SUCCEEDED, desc.state)
