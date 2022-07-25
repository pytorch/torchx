# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, cast, Iterable, Iterator, List, Optional, Type
from unittest import TestCase
from unittest.mock import patch

from torchx.schedulers import get_scheduler_factories
from torchx.schedulers.api import AppDryRunInfo, DescribeAppResponse
from torchx.schedulers.ray.ray_common import RayActor
from torchx.schedulers.ray_scheduler import has_ray
from torchx.specs import AppDef, Resource, Role, runopts


if has_ray():
    import ray
    from ray.cluster_utils import Cluster
    from torchx.schedulers.ray import ray_driver
    from torchx.schedulers.ray_scheduler import (
        _logger,
        RayJob,
        RayOpts,
        RayScheduler,
        serialize,
    )

    class RaySchedulerRegistryTest(TestCase):
        def test_get_schedulers_returns_ray_scheduler(self) -> None:
            schedulers = get_scheduler_factories()

            self.assertIn("ray", schedulers)

            scheduler = schedulers["ray"]("test_session")

            self.assertIsInstance(scheduler, RayScheduler)

            ray_scheduler = cast(RayScheduler, scheduler)

            self.assertEqual(ray_scheduler.backend, "ray")
            self.assertEqual(ray_scheduler.session_name, "test_session")

    class RaySchedulerTest(TestCase):
        def setUp(self) -> None:
            self._scripts = ["dummy1.py", "dummy2.py"]

            self.tempdir = tempfile.TemporaryDirectory()

            self._app_def = AppDef(
                name="dummy_app",
                roles=[
                    Role(
                        name="dummy_role1",
                        image=self.tempdir.name,
                        entrypoint="dummy_entrypoint1",
                        args=["arg1", self._scripts[0], "arg2"],
                        num_replicas=3,
                        env={"dummy_env": "dummy_value"},
                        resource=Resource(cpu=2, gpu=3, memMB=0),
                    ),
                    Role(
                        name="dummy_role2",
                        image=self.tempdir.name,
                        entrypoint="dummy_entrypoint2",
                        args=["arg3", "arg4", self._scripts[1]],
                    ),
                ],
            )

            self._run_cfg = RayOpts(
                {
                    "cluster_config_file": "dummy_file",
                    "cluster_name": "dummy_name",
                    "working_dir": None,
                    "requirements": None,
                }
            )

            self._scheduler = RayScheduler("test_session")

            self._isfile_patch = patch("torchx.schedulers.ray_scheduler.os.path.isfile")

            self._mock_isfile = self._isfile_patch.start()
            self._mock_isfile.return_value = True

        def tearDown(self) -> None:
            self.tempdir.cleanup()
            self._isfile_patch.stop()

        def test_init_sets_session_and_backend_name(self) -> None:
            self.assertEqual(self._scheduler.backend, "ray")
            self.assertEqual(self._scheduler.session_name, "test_session")

        def test_run_opts_returns_expected_options(self) -> None:
            opts: runopts = self._scheduler.run_opts()

            @dataclass
            class Option:
                name: str
                opt_type: Type
                is_required: bool = False
                default: Any = None

            def assert_option(expected_opt: Option) -> None:
                opt = opts.get(expected_opt.name)

                self.assertIsNotNone(opt)

                self.assertEqual(opt.opt_type, expected_opt.opt_type)
                self.assertEqual(opt.is_required, expected_opt.is_required)

                if expected_opt.default is None:
                    self.assertIsNone(opt.default)
                else:
                    self.assertEqual(opt.default, expected_opt.default)

            expected_opts = [
                Option("cluster_config_file", str, is_required=False),
                Option("cluster_name", str),
                Option("dashboard_address", str, default="127.0.0.1:8265"),
                Option("requirements", str, is_required=False),
            ]

            self.assertEqual(len(opts), len(expected_opts))

            for expected_opt in expected_opts:
                assert_option(expected_opt)

        def test_validate_does_not_raise_error_and_does_not_log_warning(self) -> None:
            with self.assertLogs(_logger, "WARNING") as cm:
                self._scheduler._validate(self._app_def, scheduler="ray")

                _logger.warning("dummy log")

            self.assertEqual(len(cm.records), 1)

        def test_validate_raises_error_if_backend_name_is_not_ray(self) -> None:
            with self.assertRaisesRegex(
                ValueError,
                r"^An unknown scheduler backend 'dummy' has been passed to the Ray scheduler.$",
            ):
                self._scheduler._validate(self._app_def, scheduler="dummy")

        @contextmanager
        def _assert_log_message(self, level: str, msg: str) -> Iterator[None]:
            with self.assertLogs(_logger) as cm:
                yield

            self.assertEqual(len(cm.records), 1)

            log_record = cm.records[0]

            self.assertEqual(log_record.levelname, level)
            self.assertEqual(log_record.message, msg)

        def test_validate_warns_when_app_def_contains_metadata(self) -> None:
            self._app_def.metadata["dummy_key"] = "dummy_value"

            with self._assert_log_message(
                "WARNING", "The Ray scheduler does not use metadata information."
            ):
                self._scheduler._validate(self._app_def, scheduler="ray")

        def test_validate_warns_when_role_contains_resource_capability(self) -> None:
            self._app_def.roles[1].resource.capabilities["dummy_cap1"] = 1
            self._app_def.roles[1].resource.capabilities["dummy_cap2"] = 2

            with self._assert_log_message(
                "WARNING",
                "The Ray scheduler does not support custom resource capabilities.",
            ):
                self._scheduler._validate(self._app_def, scheduler="ray")

        def test_validate_warns_when_role_contains_port_map(self) -> None:
            self._app_def.roles[1].port_map["dummy_map1"] = 1
            self._app_def.roles[1].port_map["dummy_map2"] = 2

            with self._assert_log_message(
                "WARNING", "The Ray scheduler does not support port mapping."
            ):
                self._scheduler._validate(self._app_def, scheduler="ray")

        def test_submit_dryrun_raises_error_if_cluster_config_file_is_not_str(
            self,
        ) -> None:
            # pyre-fixme: Expects string type
            self._run_cfg["cluster_config_file"] = 1

            with self.assertRaisesRegex(
                ValueError,
                r"^The cluster configuration file must be a YAML file.$",
            ):
                self._scheduler._submit_dryrun(self._app_def, self._run_cfg)

        def test_submit_dryrun_raises_error_if_cluster_config_file_is_not_found(
            self,
        ) -> None:
            self._mock_isfile.return_value = False

            with self.assertRaisesRegex(
                ValueError,
                r"^The cluster configuration file must be a YAML file.$",
            ):
                self._scheduler._submit_dryrun(self._app_def, self._run_cfg)

        # pyre-ignore[2]: Parameter `value` must have a type other than `Any`
        def _assert_config_value(self, name: str, value: Any, type_name: str) -> None:
            # pyre-fixme: TypedDict indexes by string literal
            self._run_cfg[name] = value

            with self.assertRaisesRegex(
                TypeError,
                rf"^The configuration value '{name}' must be of type {type_name}.$",
            ):
                self._scheduler._submit_dryrun(self._app_def, self._run_cfg)

        def _assert_submit_dryrun_constructs_job_definition(self) -> None:
            run_info = self._scheduler._submit_dryrun(self._app_def, self._run_cfg)

            job = run_info.request

            self.assertTrue(job.app_id.startswith(self._app_def.name))
            self.assertGreater(len(job.app_id), len(self._app_def.name))

            self.assertEqual(
                job.cluster_config_file, self._run_cfg.get("cluster_config_file")
            )
            self.assertEqual(job.cluster_name, self._run_cfg.get("cluster_name"))

            actor_roles = []
            for role in self._app_def.roles:
                actor_roles += [role] * role.num_replicas

            self.assertEqual(len(job.actors), len(actor_roles))

            for actor, role in zip(job.actors, actor_roles):
                self.assertEqual(actor.name, role.name)
                self.assertEqual(actor.command, [role.entrypoint] + role.args)
                self.assertEqual(actor.env, role.env)
                self.assertEqual(actor.num_cpus, max(1, role.resource.cpu))
                self.assertEqual(actor.num_gpus, max(0, role.resource.gpu))

        def test_submit_dryrun_constructs_job_definition(self) -> None:
            self._assert_submit_dryrun_constructs_job_definition()

            self._run_cfg["cluster_name"] = None
            self._run_cfg["working_dir"] = None
            self._run_cfg["requirements"] = None

            self._assert_submit_dryrun_constructs_job_definition()

        def test_submit_dryrun_constructs_actor_command(self) -> None:
            run_info = self._scheduler._submit_dryrun(self._app_def, self._run_cfg)

            job = run_info.request

            self.assertEqual(
                job.actors[0].command,
                ["dummy_entrypoint1", "arg1", "dummy1.py", "arg2"],
            )

        def test_no_dir(self) -> None:
            app = AppDef(
                name="dummy_app",
                roles=[
                    Role(
                        name="dummy_role1",
                        image="invalid_path",
                    ),
                ],
            )
            with self.assertRaisesRegex(
                RuntimeError, "Role image must be a valid directory, got: invalid_path"
            ):
                self._scheduler._submit_dryrun(app, cfg={})

        def test_requirements(self) -> None:
            with tempfile.TemporaryDirectory() as path:
                reqs = os.path.join(path, "requirements.txt")
                with open(reqs, "w") as f:
                    f.write("asdf")

                app = AppDef(
                    name="app",
                    roles=[
                        Role(
                            name="role",
                            image=path,
                        ),
                    ],
                )
                req = self._scheduler._submit_dryrun(app, cfg={})
                job = req.request
                self.assertEqual(job.requirements, reqs)

        def test_parse_app_id(self) -> None:
            test_addr_appid = [
                (
                    "0.0.0.0:1234-app_id",
                    "0.0.0.0:1234",
                    "app_id",
                ),  # (full address, address:port, app_id)
                ("addr-of-cluster:1234-app-id", "addr-of-cluster:1234", "app-id"),
                ("www.test.com:1234-app:id", "www.test.com:1234", "app:id"),
                ("foo", "foo", ""),
                ("foo-bar-bar", "foo", "bar-bar"),
            ]
            for test_example, addr, app_id in test_addr_appid:
                parsed_addr, parsed_appid = self._scheduler._parse_app_id(test_example)
                self.assertEqual(parsed_addr, addr)
                self.assertEqual(parsed_appid, app_id)

        def test_list_throws_without_address(self) -> None:
            with self.assertRaisesRegex(
                Exception, "RAY_ADDRESS env variable is expected"
            ):
                self._scheduler.list()

    class RayClusterSetup:
        _instance = None
        _cluster = None

        def __new__(cls):
            if cls._instance is None:
                cls._instance = super(RayClusterSetup, cls).__new__(cls)
                ray.shutdown()
                cls._cluster = Cluster(
                    initialize_head=True,
                    head_node_args={
                        "num_cpus": 2,
                    },
                )
                cls._cluster.connect()
                cls._cluster.add_node()
                cls.reference_count = 0
            return cls._instance

        @property
        def workers(self):
            return list(self._cluster.worker_nodes)

        def add_node(cls):
            # add 1 node with 2 cpus to the cluster
            cls._cluster.add_node(num_cpus=2)

        def remove_node(cls):
            # randomly remove 1 node from the cluster
            cls._cluster.remove_node(cls.workers[0])

        def increment_reference(cls) -> None:
            cls.reference_count += 1
            if cls.reference_count <= 0:
                raise AssertionError("not correctly implemented")

        def decrement_reference(cls) -> None:
            cls.reference_count -= 1
            if cls.reference_count == 0:
                cls.teardown_ray_cluster()

        def teardown_ray_cluster(cls) -> None:
            # pyre-fixme[16]: Module `worker` has no attribute `shutdown`.
            ray.shutdown()
            cls._cluster.shutdown()
            del os.environ["RAY_ADDRESS"]

    class RayDriverTest(TestCase):
        def test_actors_serialize(self) -> None:
            assert ray.cluster_resources()["CPU"] == 4
            actor1 = RayActor(
                name="test_actor_1", command=["python", "1", "2"], env={"fake": "1"}
            )
            actor2 = RayActor(
                name="test_actor_2", command=["python", "3", "4"], env={"fake": "2"}
            )
            actors = [actor1, actor2]
            current_dir = os.path.dirname(os.path.realpath(__file__))
            serialize(actors, current_dir)

            loaded_actor = ray_driver.load_actor_json(
                os.path.join(current_dir, "actors.json")
            )
            self.assertEqual(loaded_actor, actors)

        def test_command_actor_setup(self) -> None:
            """Create enough placement groups before the creation of the command actors"""
            ray_cluster_setup = RayClusterSetup()
            ray_cluster_setup.increment_reference()
            assert ray.cluster_resources()["CPU"] == 4

            actor1 = RayActor(
                name="test_actor_1", command=["python", "1", "2"], env={"fake": "1"}
            )
            actor2 = RayActor(
                name="test_actor_2", command=["python", "3", "4"], env={"fake": "2"}
            )
            actors = [actor1, actor2]

            pg = ray_driver.create_placement_group(actors)
            command_actors = ray_driver.create_command_actors(actors, pg)

            self.assertEqual(len(command_actors), 2)
            ray.util.placement_group.remove_placement_group(pg)
            ray_cluster_setup.decrement_reference()

        def test_placement_group_creation(self) -> None:
            """Test the logic in ray driver's main loop, if the placement group can be created
            during the execution of other actors
            """
            from ray.util.placement_group import PlacementGroup

            ray_cluster_setup = RayClusterSetup()
            ray_cluster_setup.increment_reference()
            ray_cluster_setup.remove_node()  # remove one node, now there should be only one node
            assert ray.cluster_resources()["CPU"] == 2

            actor1 = RayActor(
                name="test_actor_1",
                command=["python", "-c" 'print("test_actor_1")'],
                env={"fake": "1"},
            )
            actor2 = RayActor(
                name="test_actor_2",
                command=["python", "-c" 'print("test_actor_2")'],
                env={"fake": "2"},
            )

            pg1 = ray_driver.create_placement_group(
                [
                    actor1,
                ]
            )
            command_actor1 = ray_driver.create_command_actors([actor1], pg1)[0]

            active_workers = [
                command_actor1.exec_module.remote(),
            ]  # pyre-ignore
            pg2 = ray_driver.create_placement_group_async(
                [
                    actor2,
                ]
            )
            active_workers.append(
                pg2.ready()
            )  # before we add a node, pg2 should be always pending

            completed_workers, active_workers = ray.wait(
                active_workers
            )  # wait for actor in pg1 to be finished
            self.assertEqual(len(completed_workers), 1)
            self.assertEqual(len(active_workers), 1)

            ray_cluster_setup.add_node()  # now it should have enough cpus to create pg2
            completed_workers, active_workers = ray.wait(
                active_workers
            )  # wait for actor in pg2 to be finished
            self.assertEqual(len(completed_workers), 1)
            self.assertEqual(len(active_workers), 0)
            self.assertTrue(isinstance(ray.get(active_workers[0]), PlacementGroup))

            ray.util.placement_group.remove_placement_group(pg1)
            ray.util.placement_group.remove_placement_group(pg2)
            ray_cluster_setup.decrement_reference()

    class RayIntegrationTest(TestCase):
        def test_ray_cluster(self) -> None:
            assert ray.cluster_resources()["CPU"] == 4
            ray_cluster_setup = RayClusterSetup()
            ray_cluster_setup.increment_reference()
            ray_scheduler = self.setup_ray_cluster()
            # pyre-fixme[16]: Module `worker` has no attribute `is_initialized`.
            self.assertTrue(ray.is_initialized())

            job_id = self.schedule_ray_job(ray_scheduler)
            self.assertIsNotNone(job_id)

            ray_scheduler.wait_until_finish(job_id, 100)

            logs = self.check_logs(ray_scheduler=ray_scheduler, app_id=job_id)
            print(logs)
            self.assertIsNotNone(logs)

            status = self.describe(ray_scheduler, job_id)
            self.assertIsNotNone(status)

            app_handles = self.list(ray_scheduler)
            self.assertEqual(app_handles, [job_id])

            ray_cluster_setup.decrement_reference()

        def setup_ray_cluster(self) -> RayScheduler:
            ray_scheduler = RayScheduler(session_name="test")
            return ray_scheduler

        def schedule_ray_job(
            self, ray_scheduler: RayScheduler, app_id: str = "123"
        ) -> str:
            current_dir = os.path.dirname(os.path.realpath(__file__))
            actors = [
                RayActor(
                    name="ddp",
                    num_cpus=2,
                    command=os.path.join(current_dir, "train.py"),
                ),
                RayActor(
                    name="ddp",
                    num_cpus=2,
                    command=os.path.join(current_dir, "train.py"),
                ),
            ]

            ray_job = RayJob(
                app_id=app_id,
                dashboard_address="127.0.0.1:8265",
                actors=actors,
                working_dir=current_dir,
            )
            app_info = AppDryRunInfo(ray_job, repr)
            job_id = ray_scheduler.schedule(app_info)
            return job_id

        def describe(
            self, ray_scheduler: RayScheduler, app_id: str = "123"
        ) -> Optional[DescribeAppResponse]:
            return ray_scheduler.describe(app_id)

        def check_logs(
            self, ray_scheduler: RayScheduler, app_id: str = "123"
        ) -> Iterable[str]:
            return ray_scheduler.log_iter(app_id=app_id)

        def list(self, ray_scheduler: RayScheduler) -> List[str]:
            os.environ["RAY_ADDRESS"] = "http://127.0.0.1:8265"
            return ray_scheduler.list()
