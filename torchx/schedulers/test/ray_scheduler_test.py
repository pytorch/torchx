# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, Iterator, Type
from unittest import TestCase
from unittest.mock import patch

from torchx.schedulers.ray_scheduler import RayScheduler, _logger, has_ray
from torchx.specs import AppDef, CfgVal, Resource, Role, runopts


if has_ray():

    # TODO(aivanou): enable after 0.1.1 release
    # class RaySchedulerRegistryTest(TestCase):
    #     def test_get_schedulers_returns_ray_scheduler(self) -> None:
    #         schedulers = get_schedulers("test_session")

    #         self.assertIn("ray", schedulers)

    #         scheduler = schedulers["ray"]

    #         self.assertIsInstance(scheduler, RayScheduler)

    #         ray_scheduler = cast(RayScheduler, scheduler)

    #         self.assertEqual(ray_scheduler.backend, "ray")
    #         self.assertEqual(ray_scheduler.session_name, "test_session")

    class RaySchedulerTest(TestCase):
        def setUp(self) -> None:
            self._scripts = ["dummy1.py", "dummy2.py"]

            self._app_def = AppDef(
                name="dummy_app",
                roles=[
                    Role(
                        name="dummy_role1",
                        image="dummy_img1",
                        entrypoint="dummy_entrypoint1",
                        args=["arg1", self._scripts[0], "arg2"],
                        num_replicas=3,
                        env={"dummy_env": "dummy_value"},
                        resource=Resource(cpu=2, gpu=3, memMB=0),
                    ),
                    Role(
                        name="dummy_role2",
                        image="dummy_img2",
                        entrypoint="dummy_entrypoint2",
                        args=["arg3", "arg4", self._scripts[1]],
                    ),
                ],
            )

            self._run_cfg: Dict[str, CfgVal] = {
                "cluster_config_file": "dummy_file",
                "cluster_name": "dummy_name",
                "copy_scripts": True,
                "copy_script_dirs": True,
                "verbose": True,
            }

            self._scheduler = RayScheduler("test_session")

            self._isfile_patch = patch("torchx.schedulers.ray_scheduler.os.path.isfile")

            self._mock_isfile = self._isfile_patch.start()
            self._mock_isfile.return_value = True

        def tearDown(self) -> None:
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
                Option("cluster_config_file", str, is_required=True),
                Option("cluster_name", str),
                Option("copy_scripts", bool, default=False),
                Option("copy_script_dirs", bool, default=False),
                Option("verbose", bool, default=False),
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
            self._run_cfg[name] = value

            with self.assertRaisesRegex(
                TypeError,
                rf"^The configuration value '{name}' must be of type {type_name}.$",
            ):
                self._scheduler._submit_dryrun(self._app_def, self._run_cfg)

        def test_submit_dryrun_raises_error_if_cluster_name_is_not_str(self) -> None:
            self._assert_config_value("cluster_name", 1, "str")

        def test_submit_dryrun_raises_error_if_copy_scripts_is_not_bool(self) -> None:
            self._assert_config_value("copy_scripts", "dummy_value", "bool")

        def test_submit_dryrun_raises_error_if_copy_script_dirs_is_not_bool(
            self,
        ) -> None:
            self._assert_config_value("copy_script_dirs", "dummy_value", "bool")

        def test_submit_dryrun_raises_error_if_verbose_is_not_bool(self) -> None:
            self._assert_config_value("verbose", "dummy_value", "bool")

        def _assert_submit_dryrun_constructs_job_definition(self) -> None:
            run_info = self._scheduler._submit_dryrun(self._app_def, self._run_cfg)

            job = run_info.request

            self.assertTrue(job.app_id.startswith(self._app_def.name))
            self.assertGreater(len(job.app_id), len(self._app_def.name))

            self.assertEqual(
                job.cluster_config_file, self._run_cfg.get("cluster_config_file")
            )
            self.assertEqual(job.cluster_name, self._run_cfg.get("cluster_name"))
            self.assertEqual(
                job.copy_scripts, self._run_cfg.get("copy_scripts") or False
            )
            self.assertEqual(
                job.copy_script_dirs, self._run_cfg.get("copy_script_dirs") or False
            )
            self.assertEqual(job.verbose, self._run_cfg.get("verbose") or False)

            for actor, role in zip(job.actors, self._app_def.roles):
                self.assertEqual(actor.name, role.name)
                self.assertEqual(actor.command, " ".join([role.entrypoint] + role.args))
                self.assertEqual(actor.env, role.env)
                self.assertEqual(actor.num_replicas, max(1, role.num_replicas))
                self.assertEqual(actor.num_cpus, max(1, role.resource.cpu))
                self.assertEqual(actor.num_gpus, max(0, role.resource.gpu))

            if job.copy_scripts:
                self.assertEqual(job.scripts, set(self._scripts))
            else:
                self.assertEqual(job.scripts, set())

        def test_submit_dryrun_constructs_job_definition(self) -> None:
            self._assert_submit_dryrun_constructs_job_definition()

            self._run_cfg["cluster_name"] = None
            self._run_cfg["copy_scripts"] = False
            self._run_cfg["copy_script_dirs"] = False
            self._run_cfg["verbose"] = None

            self._assert_submit_dryrun_constructs_job_definition()

        def test_submit_dryrun_constructs_actor_command(self) -> None:
            run_info = self._scheduler._submit_dryrun(self._app_def, self._run_cfg)

            job = run_info.request

            self.assertEqual(
                job.actors[0].command, "dummy_entrypoint1 arg1 dummy1.py arg2"
            )
