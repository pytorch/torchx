#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import datetime
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from pyre_extensions import none_throws
from torchx.runner import Runner, get_runner
from torchx.schedulers.api import DescribeAppResponse
from torchx.schedulers.local_scheduler import (
    LocalDirectoryImageProvider,
    LocalScheduler,
)
from torchx.schedulers.test.test_util import write_shell_script
from torchx.specs.api import (
    AppDef,
    AppState,
    Resource,
    Role,
    UnknownAppException,
)
from torchx.specs.finder import ComponentNotFoundException


GET_SCHEDULERS = "torchx.runner.api.get_schedulers"


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

        self.scheduler = LocalScheduler(
            SESSION_NAME, image_provider_class=LocalDirectoryImageProvider
        )
        self.cfg = {}

        # resource ignored for local scheduler; adding as an example

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def test_validate_no_roles(self, _) -> None:
        with Runner("test", schedulers={"local_dir": self.scheduler}) as runner:
            with self.assertRaises(ValueError):
                app = AppDef("no roles")
                runner.run(app, scheduler="local_dir")

    def test_validate_no_resource(self, _) -> None:
        with Runner("test", schedulers={"local_dir": self.scheduler}) as runner:
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
        with Runner("test", schedulers={"local_dir": self.scheduler}) as runner:
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

        with Runner(
            name=SESSION_NAME,
            schedulers={"local_dir": self.scheduler},
        ) as runner:
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
            schedulers={"local_dir": scheduler_mock},
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

    def test_describe(self, _) -> None:
        with Runner(
            name=SESSION_NAME, schedulers={"local_dir": self.scheduler}
        ) as runner:
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

    def test_list(self, _) -> None:
        with Runner(
            name=SESSION_NAME,
            schedulers={"local_dir": self.scheduler},
        ) as runner:
            role = Role(
                name="touch",
                image=self.test_dir,
                resource=resource.SMALL,
                entrypoint="sleep.sh",
                args=["1"],
            )
            app = AppDef("sleeper", roles=[role])

            num_apps = 4

            for _ in range(num_apps):
                # since this test validates the list() API,
                # we do not wait for the apps to finish so run the apps
                # in managed mode so that the local scheduler reaps the apps on exit
                runner.run(app, scheduler="local_dir")

            apps = runner.list()
            self.assertEqual(num_apps, len(apps))

    def test_evict_non_existent_app(self, _) -> None:
        # tests that apps previously run with this session that are finished and eventually
        # removed by the scheduler also get removed from the session after a status() API has been
        # called on the app

        scheduler = LocalScheduler(
            SESSION_NAME, cache_size=1, image_provider_class=LocalDirectoryImageProvider
        )
        with Runner(
            name=SESSION_NAME,
            schedulers={"local_dir": scheduler},
        ) as runner:
            test_file = os.path.join(self.test_dir, "test_file")
            role = Role(
                name="touch",
                image=self.test_dir,
                resource=resource.SMALL,
                entrypoint="touch.sh",
                args=[test_file],
            )
            app = AppDef("touch_test_file", roles=[role])

            # local scheduler was setup with a cache size of 1
            # run the same app twice (the first will be removed from the scheduler's cache)
            # then validate that the first one will drop from the session's app cache as well
            app_id1 = runner.run(app, scheduler="local_dir", cfg=self.cfg)
            runner.wait(app_id1, wait_interval=0.1)

            app_id2 = runner.run(app, scheduler="local_dir", cfg=self.cfg)
            runner.wait(app_id2, wait_interval=0.1)

            apps = runner.list()

            self.assertEqual(1, len(apps))
            self.assertFalse(app_id1 in apps)
            self.assertTrue(app_id2 in apps)

    def test_status(self, _) -> None:
        with Runner(
            name=SESSION_NAME,
            schedulers={"local_dir": self.scheduler},
        ) as runner:
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
            runner.stop(app_handle)
            app_status = none_throws(runner.status(app_handle))
            self.assertEqual(AppState.CANCELLED, app_status.state)

    def test_status_unknown_app(self, _) -> None:
        with Runner(
            name=SESSION_NAME,
            schedulers={"local_dir": self.scheduler},
        ) as runner:
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
            name="test_ui_url_session", schedulers={"local_dir": mock_scheduler}
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
            name="test_structured_msg", schedulers={"local_dir": mock_scheduler}
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
        with Runner(
            name=SESSION_NAME,
            schedulers={"local_dir": self.scheduler},
        ) as runner:
            self.assertIsNone(
                runner.wait(
                    "local_dir://test_session/unknown_app_id", wait_interval=0.1
                )
            )
            self.assertIsNone(
                runner.wait("local_dir://another_session/some_app", wait_interval=0.1)
            )

    def test_stop(self, _) -> None:
        with Runner(
            name=SESSION_NAME,
            schedulers={"local_dir": self.scheduler},
        ) as runner:
            self.assertIsNone(runner.stop("local_dir://test_session/unknown_app_id"))

    def test_log_lines_unknown_app(self, _) -> None:
        with Runner(
            name=SESSION_NAME,
            schedulers={"local_dir": self.scheduler},
        ) as runner:
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
            schedulers={"local_dir": scheduler_mock},
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

    @patch("json.dumps")
    def test_get_schedulers(self, json_dumps_mock: MagicMock, _) -> None:
        local_dir_sched_mock = MagicMock()
        json_dumps_mock.return_value = "{}"
        local_sched_mock = MagicMock()
        schedulers = {"local_dir": local_dir_sched_mock, "local": local_sched_mock}
        with Runner(name="test_session", schedulers=schedulers) as runner:
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

    def test_run_from_module(self, _) -> None:
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
        schedulers = {"local_dir": local_sched_mock, "local": local_sched_mock}
        with Runner(name="test_session", schedulers=schedulers) as runner:
            component_path = get_full_path("distributed.py")
            with patch.object(runner, "run"):
                with self.assertRaises(ComponentNotFoundException):
                    runner.run_component(
                        f"{component_path}:unknown_function", [], "local"
                    )

    def test_runner_context_manager(self, _) -> None:
        mock_scheduler = MagicMock()
        with patch(GET_SCHEDULERS, return_value={"local_dir": mock_scheduler}):
            with get_runner() as runner:
                pass
        mock_scheduler.close.assert_called_once()

    def test_runner_context_manager_with_error(self, _) -> None:
        mock_scheduler = MagicMock()
        with patch(GET_SCHEDULERS, return_value={"local_dir": mock_scheduler}):
            with self.assertRaisesRegex(RuntimeError, "foobar"):
                with get_runner() as runner:
                    raise RuntimeError("foobar")

    def test_runner_try_catch(self, _) -> None:
        mock_scheduler = MagicMock()
        with patch(GET_SCHEDULERS, return_value={"local_dir": mock_scheduler}):
            runner = get_runner()
            try:
                num_schedulers = len(runner._schedulers)
            finally:
                runner.close()

        mock_scheduler.close.assert_called_once()

        # can call close twice
        runner.close()

    def test_get_default_runner(self, _) -> None:
        runner = get_runner()
        self.assertEqual("torchx", runner._name)
