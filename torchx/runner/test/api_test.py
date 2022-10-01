#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import os
import shutil
import tempfile
import unittest
from contextlib import contextmanager
from typing import Generator, List, Mapping, Optional
from unittest.mock import MagicMock, patch

from torchx.runner import get_runner, Runner
from torchx.schedulers.api import DescribeAppResponse, ListAppResponse, Scheduler
from torchx.schedulers.local_scheduler import (
    LocalDirectoryImageProvider,
    LocalScheduler,
)
from torchx.schedulers.test.test_util import write_shell_script
from torchx.specs import AppDryRunInfo, CfgVal
from torchx.specs.api import AppDef, AppState, Resource, Role, UnknownAppException
from torchx.specs.finder import ComponentNotFoundException

from torchx.util.types import none_throws
from torchx.workspace import Workspace


GET_SCHEDULER_FACTORIES = "torchx.runner.api.get_scheduler_factories"


class resource:
    SMALL = Resource(cpu=1, gpu=0, memMB=1024)
    MEDIUM = Resource(cpu=4, gpu=0, memMB=(4 * 1024))
    LARGE = Resource(cpu=16, gpu=0, memMB=(16 * 1024))


SESSION_NAME = "test_session"


def get_full_path(name: str) -> str:
    return os.path.join(os.path.dirname(__file__), "resource", name)


@patch("torchx.runner.api.log_event")
class RunnerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp("RunnerTest")

        write_shell_script(self.test_dir, "touch.sh", ["touch $1"])
        write_shell_script(self.test_dir, "fail.sh", ["exit 1"])
        write_shell_script(self.test_dir, "sleep.sh", ["sleep $1"])

        self.cfg = {}

        # resource ignored for local scheduler; adding as an example

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    @contextmanager
    def get_runner(self) -> Generator[Runner, None, None]:
        with Runner(
            SESSION_NAME,
            scheduler_factories={"local_dir": LocalScheduler},
            scheduler_params={
                "image_provider_class": LocalDirectoryImageProvider,
            },
        ) as runner:
            yield runner

    def test_validate_no_roles(self, _) -> None:
        with self.get_runner() as runner:
            with self.assertRaises(ValueError):
                app = AppDef("no roles")
                runner.run(app, scheduler="local_dir")

    def test_validate_no_resource(self, _) -> None:
        with self.get_runner() as runner:
            with self.assertRaises(ValueError):
                role = Role(
                    "no resource",
                    image="no_image",
                    entrypoint="echo",
                    args=["hello_world"],
                )
                app = AppDef("no resource", roles=[role])
                runner.run(app, scheduler="local_dir")

    def test_validate_invalid_replicas(self, _) -> None:
        with self.get_runner() as runner:
            with self.assertRaises(ValueError):
                role = Role(
                    "invalid replicas",
                    image="torch",
                    entrypoint="echo",
                    args=["hello_world"],
                    num_replicas=0,
                    resource=Resource(cpu=1, gpu=0, memMB=500),
                )
                app = AppDef("invalid replicas", roles=[role])
                runner.run(app, scheduler="local_dir")

    def test_run(self, _) -> None:
        test_file = os.path.join(self.test_dir, "test_file")

        with self.get_runner() as runner:
            self.assertEqual(1, len(runner.scheduler_backends()))
            role = Role(
                name="touch",
                image=self.test_dir,
                resource=resource.SMALL,
                entrypoint="touch.sh",
                args=[test_file],
            )
            app = AppDef("name", roles=[role])

            app_handle = runner.run(app, scheduler="local_dir", cfg=self.cfg)
            app_status = none_throws(runner.wait(app_handle, wait_interval=0.1))
            self.assertEqual(AppState.SUCCEEDED, app_status.state)

    def test_dryrun(self, _) -> None:
        scheduler_mock = MagicMock()
        with Runner(
            name=SESSION_NAME,
            scheduler_factories={"local_dir": lambda name: scheduler_mock},
        ) as runner:
            role = Role(
                name="touch",
                image=self.test_dir,
                resource=resource.SMALL,
                entrypoint="echo",
                args=["hello world"],
            )
            app = AppDef("name", roles=[role])
            runner.dryrun(app, "local_dir", cfg=self.cfg)
            scheduler_mock.submit_dryrun.assert_called_once_with(app, self.cfg)
            scheduler_mock._validate.assert_called_once()

    def test_dryrun_env_variables(self, _) -> None:
        scheduler_mock = MagicMock()
        with Runner(
            name=SESSION_NAME,
            scheduler_factories={"local_dir": lambda name: scheduler_mock},
        ) as runner:
            role1 = Role(
                name="echo1",
                image=self.test_dir,
                resource=resource.SMALL,
                entrypoint="echo",
                args=["hello world"],
            )
            role2 = Role(
                name="echo2",
                image=self.test_dir,
                resource=resource.SMALL,
                entrypoint="echo",
                args=["hello world"],
            )
            app = AppDef("name", roles=[role1, role2])
            runner.dryrun(app, "local_dir", cfg=self.cfg)
            for role in app.roles:
                self.assertEqual(
                    role.env["TORCHX_JOB_ID"],
                    "local_dir://" + SESSION_NAME + "/${app_id}",
                )

    def test_dryrun_trackers_parent_run_id_as_paramenter(self, _) -> None:
        scheduler_mock = MagicMock()
        expected_parent_run_id = "123"
        with Runner(
            name=SESSION_NAME,
            scheduler_factories={"local_dir": lambda name: scheduler_mock},
        ) as runner:
            role1 = Role(
                name="echo1",
                image=self.test_dir,
                resource=resource.SMALL,
                entrypoint="echo",
                args=["hello world"],
            )
            role2 = Role(
                name="echo2",
                image=self.test_dir,
                resource=resource.SMALL,
                entrypoint="echo",
                args=["hello world"],
            )
            app = AppDef("name", roles=[role1, role2])
            runner.dryrun(
                app, "local_dir", cfg=self.cfg, parent_run_id=expected_parent_run_id
            )
            for role in app.roles:
                self.assertEqual(
                    role.env["TORCHX_PARENT_RUN_ID"],
                    expected_parent_run_id,
                )

    @patch("torchx.runner.api._get_configured_trackers")
    def test_dryrun_setup_trackers(self, config_trackers_mock: MagicMock, _) -> None:
        config_trackers_mock.return_value = {
            "my_tracker1": "manifold://config1.txt",
            "my_tracker2": "manifold://config2.txt",
        }
        scheduler_mock = MagicMock()
        expected_trackers = "my_tracker1,my_tracker2"
        expected_tracker1_config = "manifold://config1.txt"
        expected_tracker2_config = "manifold://config2.txt"

        with Runner(
            name=SESSION_NAME,
            scheduler_factories={"local_dir": lambda name: scheduler_mock},
        ) as runner:
            role1 = Role(
                name="echo1",
                image=self.test_dir,
                resource=resource.SMALL,
                entrypoint="echo",
                args=["hello world"],
            )
            role2 = Role(
                name="echo2",
                image=self.test_dir,
                resource=resource.SMALL,
                entrypoint="echo",
                args=["hello world"],
            )
            app = AppDef("name", roles=[role1, role2])
            runner.dryrun(app, "local_dir", cfg=self.cfg, parent_run_id="999")
            for role in app.roles:
                self.assertEqual(
                    role.env["TORCHX_TRACKERS"],
                    expected_trackers,
                )
                self.assertEqual(
                    role.env["TORCHX_TRACKER_MY_TRACKER1_CONFIG"],
                    expected_tracker1_config,
                )
                self.assertEqual(
                    role.env["TORCHX_TRACKER_MY_TRACKER2_CONFIG"],
                    expected_tracker2_config,
                )

    @patch.dict(
        os.environ,
        {
            "TORCHX_TRACKERS": "my_tracker1,my_tracker2",
            "TORCHX_TRACKER_MY_TRACKER1_CONFIG": "manifold://config1.txt",
            "TORCHX_TRACKER_MY_TRACKER2_CONFIG": "manifold://config2.txt",
        },
    )
    def test_dryrun_setup_trackers_as_env_variable(self, _) -> None:
        scheduler_mock = MagicMock()
        expected_trackers = "my_tracker1,my_tracker2"
        expected_tracker1_config = "manifold://config1.txt"
        expected_tracker2_config = "manifold://config2.txt"

        with Runner(
            name=SESSION_NAME,
            scheduler_factories={"local_dir": lambda name: scheduler_mock},
        ) as runner:
            role1 = Role(
                name="echo1",
                image=self.test_dir,
                resource=resource.SMALL,
                entrypoint="echo",
                args=["hello world"],
            )
            role2 = Role(
                name="echo2",
                image=self.test_dir,
                resource=resource.SMALL,
                entrypoint="echo",
                args=["hello world"],
            )
            app = AppDef("name", roles=[role1, role2])
            runner.dryrun(app, "local_dir", cfg=self.cfg, parent_run_id="999")
            for role in app.roles:
                self.assertEqual(
                    role.env["TORCHX_TRACKERS"],
                    expected_trackers,
                )
                self.assertEqual(
                    role.env["TORCHX_TRACKER_MY_TRACKER1_CONFIG"],
                    expected_tracker1_config,
                )
                self.assertEqual(
                    role.env["TORCHX_TRACKER_MY_TRACKER2_CONFIG"],
                    expected_tracker2_config,
                )

    def test_dryrun_with_workspace(self, _) -> None:
        class TestScheduler(Scheduler, Workspace):
            def __init__(self, build_new_img: bool):
                Scheduler.__init__(self, backend="ignored", session_name="ignored")
                self.build_new_img = build_new_img

            def schedule(self, dryrun_info: AppDryRunInfo) -> str:
                pass

            def _submit_dryrun(
                self, app: AppDef, cfg: Mapping[str, CfgVal]
            ) -> AppDryRunInfo[AppDef]:
                return AppDryRunInfo(app, lambda s: s)

            def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
                pass

            def list(self) -> List[DescribeAppResponse]:
                pass

            def _cancel_existing(self, app_id: str) -> None:
                pass

            def build_workspace_and_update_role(
                self,
                role: Role,
                workspace: str,
                cfg: Mapping[str, CfgVal],
            ) -> None:
                if self.build_new_img:
                    role.image = f"{role.image}_new"

        with Runner(
            name=SESSION_NAME,
            # pyre-fixme[6]: scheduler factory type
            scheduler_factories={
                "no-build-img": lambda name: TestScheduler(build_new_img=False),
                "builds-img": lambda name: TestScheduler(build_new_img=True),
            },
        ) as runner:
            app = AppDef(
                "ignored",
                roles=[
                    Role(
                        name="sleep",
                        image="foo",
                        resource=resource.SMALL,
                        entrypoint="sleep",
                        args=["1"],
                    ),
                    Role(
                        name="sleep",
                        image="bar",
                        resource=resource.SMALL,
                        entrypoint="sleep",
                        args=["1"],
                    ),
                ],
            )
            dryruninfo = runner.dryrun(app, "no-build-img", workspace="//workspace")
            self.assertEqual("foo", dryruninfo.request.roles[0].image)
            self.assertEqual("bar", dryruninfo.request.roles[1].image)

            dryruninfo = runner.dryrun(app, "builds-img", workspace="//workspace")
            # workspace is attached to role[0] by default
            self.assertEqual("foo_new", dryruninfo.request.roles[0].image)
            self.assertEqual("bar", dryruninfo.request.roles[1].image)

    def test_describe(self, _) -> None:
        with self.get_runner() as runner:
            role = Role(
                name="sleep",
                image=self.test_dir,
                resource=resource.SMALL,
                entrypoint="sleep.sh",
                args=["60"],
            )
            app = AppDef("sleeper", roles=[role])

            app_handle = runner.run(app, scheduler="local_dir", cfg=self.cfg)
            self.assertEqual(app, runner.describe(app_handle))
            # unknown app should return None
            self.assertIsNone(runner.describe("local_dir://session1/unknown_app"))

    def test_status(self, _) -> None:
        with self.get_runner() as runner:
            role = Role(
                name="sleep",
                image=self.test_dir,
                resource=resource.SMALL,
                entrypoint="sleep.sh",
                args=["60"],
            )
            app = AppDef("sleeper", roles=[role])
            app_handle = runner.run(app, scheduler="local_dir", cfg=self.cfg)
            app_status = none_throws(runner.status(app_handle))
            self.assertEqual(AppState.RUNNING, app_status.state)
            runner.cancel(app_handle)
            app_status = none_throws(runner.status(app_handle))
            self.assertEqual(AppState.CANCELLED, app_status.state)

    def test_status_unknown_app(self, _) -> None:
        with self.get_runner() as runner:
            self.assertIsNone(runner.status("local_dir://test_session/unknown_app_id"))

    @patch("json.dumps")
    def test_status_ui_url(self, json_dumps_mock: MagicMock, _) -> None:
        app_id = "test_app"
        json_dumps_mock.return_value = "{}"
        mock_scheduler = MagicMock()
        resp = DescribeAppResponse()
        resp.ui_url = "https://foobar"
        mock_scheduler.submit.return_value = app_id
        mock_scheduler.describe.return_value = resp

        with Runner(
            name="test_ui_url_session",
            scheduler_factories={"local_dir": lambda name: mock_scheduler},
        ) as runner:
            role = Role(
                "ignored",
                image=self.test_dir,
                resource=resource.SMALL,
                entrypoint="/bin/echo",
            )
            app_handle = runner.run(AppDef(app_id, roles=[role]), scheduler="local_dir")
            status = none_throws(runner.status(app_handle))
            self.assertEquals(resp.ui_url, status.ui_url)

    @patch("json.dumps")
    def test_status_structured_msg(self, json_dumps_mock: MagicMock, _) -> None:
        app_id = "test_app"
        json_dumps_mock.return_value = "{}"
        mock_scheduler = MagicMock()
        resp = DescribeAppResponse()
        resp.structured_error_msg = '{"message": "test error"}'
        mock_scheduler.submit.return_value = app_id
        mock_scheduler.describe.return_value = resp

        with Runner(
            name="test_structured_msg",
            scheduler_factories={"local_dir": lambda name: mock_scheduler},
        ) as runner:
            role = Role(
                "ignored",
                image=self.test_dir,
                resource=resource.SMALL,
                entrypoint="/bin/echo",
            )
            app_handle = runner.run(AppDef(app_id, roles=[role]), scheduler="local_dir")
            status = none_throws(runner.status(app_handle))
            self.assertEquals(resp.structured_error_msg, status.structured_error_msg)

    def test_wait_unknown_app(self, _) -> None:
        with self.get_runner() as runner:
            self.assertIsNone(
                runner.wait(
                    "local_dir://test_session/unknown_app_id", wait_interval=0.1
                )
            )
            self.assertIsNone(
                runner.wait("local_dir://another_session/some_app", wait_interval=0.1)
            )

    def test_cancel(self, _) -> None:
        with self.get_runner() as runner:
            self.assertIsNone(runner.cancel("local_dir://test_session/unknown_app_id"))

    def test_stop(self, _) -> None:
        with self.get_runner() as runner:
            self.assertIsNone(runner.stop("local_dir://test_session/unknown_app_id"))

    def test_log_lines_unknown_app(self, _) -> None:
        with self.get_runner() as runner:
            with self.assertRaises(UnknownAppException):
                runner.log_lines("local_dir://test_session/unknown", "trainer")

    def test_log_lines(self, _) -> None:
        app_id = "mock_app"

        scheduler_mock = MagicMock()
        scheduler_mock.describe.return_value = DescribeAppResponse(
            app_id, AppState.RUNNING
        )
        scheduler_mock.log_iter.return_value = iter(["hello", "world"])

        with Runner(
            name=SESSION_NAME,
            scheduler_factories={"local_dir": lambda name: scheduler_mock},
        ) as runner:
            role_name = "trainer"
            replica_id = 2
            regex = "QPS.*"
            since = datetime.datetime.now()
            until = datetime.datetime.now()
            lines = list(
                runner.log_lines(
                    f"local_dir://test_session/{app_id}",
                    role_name,
                    replica_id,
                    regex,
                    since,
                    until,
                )
            )

            self.assertEqual(["hello", "world"], lines)
            scheduler_mock.log_iter.assert_called_once_with(
                app_id, role_name, replica_id, regex, since, until, False, streams=None
            )

    def test_list(self, _) -> None:
        scheduler_mock = MagicMock()
        sched_list_return = [
            ListAppResponse(app_id="app_id1", state=AppState.RUNNING),
            ListAppResponse(app_id="app_id2", state=AppState.SUCCEEDED),
        ]
        scheduler_mock.list.return_value = sched_list_return
        apps_expected = [
            ListAppResponse(
                app_id="app_id1",
                app_handle="kubernetes://test_session/app_id1",
                state=AppState.RUNNING,
            ),
            ListAppResponse(
                app_id="app_id2",
                app_handle="kubernetes://test_session/app_id2",
                state=AppState.SUCCEEDED,
            ),
        ]
        with Runner(
            name=SESSION_NAME,
            scheduler_factories={"kubernetes": lambda name: scheduler_mock},
        ) as runner:
            apps = runner.list("kubernetes")
            self.assertEqual(apps, apps_expected)
            scheduler_mock.list.assert_called_once()

    @patch("json.dumps")
    def test_get_schedulers(self, json_dumps_mock: MagicMock, _) -> None:
        local_dir_sched_mock = MagicMock()
        json_dumps_mock.return_value = "{}"
        local_sched_mock = MagicMock()
        scheduler_factories = {
            "local_dir": lambda name: local_dir_sched_mock,
            "local": lambda name: local_sched_mock,
        }
        with Runner(
            name="test_session", scheduler_factories=scheduler_factories
        ) as runner:
            role = Role(
                name="sleep",
                image=self.test_dir,
                resource=resource.SMALL,
                entrypoint="sleep.sh",
                args=["60"],
            )
            app = AppDef("sleeper", roles=[role])
            runner.run(app, scheduler="local")
            local_sched_mock.submit.called_once_with(app, {})

    def test_run_from_module(self, _: str) -> None:
        runner = get_runner(name="test_session")
        app_args = ["--image", "dummy_image", "--script", "test.py"]
        with patch.object(runner, "schedule"), patch.object(
            runner, "dryrun"
        ) as dryrun_mock:
            _ = runner.run_component("dist.ddp", app_args, "local")
            args, kwargs = dryrun_mock.call_args
            actual_app = args[0]

        print(actual_app)
        self.assertEqual(actual_app.name, "test")
        self.assertEqual(1, len(actual_app.roles))
        self.assertEqual("test", actual_app.roles[0].name)

    def test_run_from_file_no_function_found(self, _) -> None:
        local_sched_mock = MagicMock()
        schedulers = {
            "local_dir": lambda name: local_sched_mock,
            "local": lambda name: local_sched_mock,
        }
        with Runner(name="test_session", scheduler_factories=schedulers) as runner:
            component_path = get_full_path("distributed.py")
            with patch.object(runner, "run"):
                with self.assertRaises(ComponentNotFoundException):
                    runner.run_component(
                        f"{component_path}:unknown_function", [], "local"
                    )

    def test_runner_context_manager(self, _) -> None:
        mock_scheduler = MagicMock()
        with patch(
            GET_SCHEDULER_FACTORIES,
            return_value={"local_dir": lambda name: mock_scheduler},
        ):
            with get_runner() as runner:
                # force schedulers to load
                runner.scheduler_run_opts("local_dir")
        mock_scheduler.close.assert_called_once()

    def test_runner_context_manager_with_error(self, _) -> None:
        mock_scheduler = MagicMock()
        with patch(
            GET_SCHEDULER_FACTORIES,
            return_value={"local_dir": lambda name: mock_scheduler},
        ):
            with self.assertRaisesRegex(RuntimeError, "foobar"):
                with get_runner() as runner:
                    raise RuntimeError("foobar")

    def test_runner_manual_close(self, _) -> None:
        mock_scheduler = MagicMock()
        with patch(
            GET_SCHEDULER_FACTORIES,
            return_value={"local_dir": lambda name: mock_scheduler},
        ):
            runner = get_runner()
            # force schedulers to load
            runner.scheduler_run_opts("local_dir")
            runner.close()

        mock_scheduler.close.assert_called_once()

        # can call close twice
        runner.close()

    def test_get_default_runner(self, _) -> None:
        runner = get_runner()
        self.assertEqual("torchx", runner._name)
