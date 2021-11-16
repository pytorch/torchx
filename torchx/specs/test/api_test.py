#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import sys
import unittest
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union
from unittest.mock import MagicMock, patch

import torchx.specs.named_resources_aws as named_resources_aws
from pyre_extensions import none_throws
from torchx.specs import named_resources
from torchx.specs.api import (
    _TERMINAL_STATES,
    MISSING,
    NULL_RESOURCE,
    AppDef,
    AppDryRunInfo,
    AppState,
    AppStatus,
    CfgVal,
    InvalidRunConfigException,
    MalformedAppHandleException,
    Resource,
    RetryPolicy,
    Role,
    _create_args_parser,
    from_function,
    get_type_name,
    macros,
    make_app_handle,
    parse_app_handle,
    runopts,
)


class AppDryRunInfoTest(unittest.TestCase):
    def test_repr(self) -> None:
        request_mock = MagicMock()
        to_string_mock = MagicMock()
        info = AppDryRunInfo(request_mock, to_string_mock)
        info.__repr__()
        self.assertEqual(request_mock, info.request)

        to_string_mock.assert_called_once_with(request_mock)


class AppDefStatusTest(unittest.TestCase):
    def test_is_terminal(self) -> None:
        for s in AppState:
            is_terminal = AppStatus(state=s).is_terminal()
            if s in _TERMINAL_STATES:
                self.assertTrue(is_terminal)
            else:
                self.assertFalse(is_terminal)

    def test_serialize(self) -> None:
        status = AppStatus(AppState.FAILED)
        serialized = repr(status)
        self.assertEqual(
            serialized,
            """AppStatus:
  msg: ''
  num_restarts: 0
  roles: []
  state: FAILED (5)
  structured_error_msg: <NONE>
  ui_url: null
""",
        )

    def test_serialize_embed_json(self) -> None:
        status = AppStatus(
            AppState.FAILED, structured_error_msg='{"message": "test error"}'
        )
        serialized = repr(status)
        self.assertEqual(
            serialized,
            """AppStatus:
  msg: ''
  num_restarts: 0
  roles: []
  state: FAILED (5)
  structured_error_msg:
    message: test error
  ui_url: null
""",
        )


class ResourceTest(unittest.TestCase):
    def test_copy_resource(self) -> None:
        old_capabilities = {"test_key": "test_value", "old_key": "old_value"}
        resource = Resource(1, 2, 3, old_capabilities)
        new_resource = Resource.copy(
            resource, test_key="test_value_new", new_key="new_value"
        )
        self.assertEqual(new_resource.cpu, 1)
        self.assertEqual(new_resource.gpu, 2)
        self.assertEqual(new_resource.memMB, 3)

        self.assertEqual(len(new_resource.capabilities), 3)
        self.assertEqual(new_resource.capabilities["old_key"], "old_value")
        self.assertEqual(new_resource.capabilities["test_key"], "test_value_new")
        self.assertEqual(new_resource.capabilities["new_key"], "new_value")
        self.assertEqual(resource.capabilities["test_key"], "test_value")

    def test_named_resources(self) -> None:
        self.assertEqual(
            named_resources_aws.aws_m5_2xlarge(), named_resources["aws_m5.2xlarge"]
        )
        self.assertEqual(
            named_resources_aws.aws_t3_medium(), named_resources["aws_t3.medium"]
        )
        self.assertEqual(
            named_resources_aws.aws_p3_2xlarge(), named_resources["aws_p3.2xlarge"]
        )
        self.assertEqual(
            named_resources_aws.aws_p3_8xlarge(), named_resources["aws_p3.8xlarge"]
        )


class RoleBuilderTest(unittest.TestCase):
    def test_defaults(self) -> None:
        default = Role("foobar", "torch")
        self.assertEqual("foobar", default.name)
        self.assertEqual("torch", default.image)
        self.assertEqual(MISSING, default.entrypoint)
        self.assertEqual({}, default.env)
        self.assertEqual([], default.args)
        self.assertEqual(NULL_RESOURCE, default.resource)
        self.assertEqual(1, default.num_replicas)
        self.assertEqual(0, default.max_retries)
        self.assertEqual(RetryPolicy.APPLICATION, default.retry_policy)
        self.assertEqual({}, default.metadata)

    def test_build_role(self) -> None:
        # runs: ENV_VAR_1=FOOBAR /bin/echo hello world
        resource = Resource(cpu=1, gpu=2, memMB=128)
        trainer = Role(
            "trainer",
            image="torch",
            entrypoint="/bin/echo",
            args=["hello", "world"],
            env={"ENV_VAR_1": "FOOBAR"},
            num_replicas=2,
            retry_policy=RetryPolicy.REPLICA,
            max_retries=5,
            resource=resource,
            port_map={"foo": 8080},
            metadata={"foo": "bar"},
        )

        self.assertEqual("trainer", trainer.name)
        self.assertEqual("torch", trainer.image)
        self.assertEqual("/bin/echo", trainer.entrypoint)
        self.assertEqual({"ENV_VAR_1": "FOOBAR"}, trainer.env)
        self.assertEqual(["hello", "world"], trainer.args)
        self.assertDictEqual({"foo": "bar"}, trainer.metadata)
        self.assertDictEqual({"foo": 8080}, trainer.port_map)
        self.assertEqual(resource, trainer.resource)
        self.assertEqual(2, trainer.num_replicas)
        self.assertEqual(5, trainer.max_retries)
        self.assertEqual(RetryPolicy.REPLICA, trainer.retry_policy)


class AppHandleTest(unittest.TestCase):
    def test_parse_malformed_app_handles(self) -> None:
        bad_app_handles = {
            "my_session/my_application_id": "missing scheduler backend",
            "local://my_session/": "missing app_id",
            "local://my_application_id": "missing session",
        }

        for handle, msg in bad_app_handles.items():
            with self.subTest(f"malformed app handle: {msg}", handle=handle):
                with self.assertRaises(MalformedAppHandleException):
                    parse_app_handle(handle)

    def test_parse(self) -> None:
        (scheduler_backend, session_name, app_id) = parse_app_handle(
            "local://my_session/my_app_id_1234"
        )
        self.assertEqual("local", scheduler_backend)
        self.assertEqual("my_session", session_name)
        self.assertEqual("my_app_id_1234", app_id)

    def test_make(self) -> None:
        app_handle = make_app_handle(
            scheduler_backend="local",
            session_name="my_session",
            app_id="my_app_id_1234",
        )
        self.assertEqual("local://my_session/my_app_id_1234", app_handle)


class AppDefTest(unittest.TestCase):
    def test_application(self) -> None:
        trainer = Role(
            "trainer",
            "test_image",
            entrypoint="/bin/sleep",
            args=["10"],
            num_replicas=2,
        )
        app = AppDef(name="test_app", roles=[trainer])
        self.assertEqual("test_app", app.name)
        self.assertEqual(1, len(app.roles))
        self.assertEqual(trainer, app.roles[0])

    def test_application_default(self) -> None:
        app = AppDef(name="test_app")
        self.assertEqual(0, len(app.roles))

    def test_getset_metadata(self) -> None:
        app = AppDef(name="test_app", metadata={"test_key": "test_value"})
        self.assertEqual("test_value", app.metadata["test_key"])
        self.assertEqual(None, app.metadata.get("non_existent"))


class RunConfigTest(unittest.TestCase):
    def get_cfg(self) -> Mapping[str, CfgVal]:
        return {
            "run_as": "root",
            "cluster_id": 123,
            "priority": 0.5,
            "preemptible": True,
        }

    def test_valid_values(self) -> None:
        cfg = self.get_cfg()

        self.assertEqual("root", cfg.get("run_as"))
        self.assertEqual(123, cfg.get("cluster_id"))
        self.assertEqual(0.5, cfg.get("priority"))
        self.assertTrue(cfg.get("preemptible"))
        self.assertIsNone(cfg.get("unknown"))

    def test_runopts_add(self) -> None:
        """
        tests for various add option variations
        does not assert anything, a successful test
        should not raise any unexpected errors
        """
        opts = runopts()
        opts.add("run_as", type_=str, help="run as user")
        opts.add("run_as_default", type_=str, help="run as user", default="root")
        opts.add("run_as_required", type_=str, help="run as user", required=True)

        with self.assertRaises(ValueError):
            opts.add(
                "run_as", type_=str, help="run as user", default="root", required=True
            )

        opts.add("priority", type_=int, help="job priority", default=10)

        with self.assertRaises(TypeError):
            opts.add("priority", type_=int, help="job priority", default=0.5)

        # this print is intentional (demonstrates the intended usecase)
        print(opts)

    def get_runopts(self) -> runopts:
        opts = runopts()
        opts.add("run_as", type_=str, help="run as user", required=True)
        opts.add("priority", type_=int, help="job priority", default=10)
        opts.add("cluster_id", type_=str, help="cluster to submit job")
        return opts

    def test_runopts_resolve_minimal(self) -> None:
        opts = self.get_runopts()
        cfg = {"run_as": "foobar"}

        resolved = opts.resolve(cfg)
        self.assertEqual("foobar", resolved.get("run_as"))
        self.assertEqual(10, resolved.get("priority"))
        self.assertIsNone(resolved.get("cluster_id"))

        # make sure original config is untouched
        self.assertEqual("foobar", cfg.get("run_as"))
        self.assertIsNone(cfg.get("priority"))
        self.assertIsNone(cfg.get("cluster_id"))

    def test_runopts_resolve_override(self) -> None:
        opts = self.get_runopts()
        cfg = {
            "run_as": "foobar",
            "priority": 20,
            "cluster_id": "test_cluster",
        }

        resolved = opts.resolve(cfg)
        self.assertEqual("foobar", resolved.get("run_as"))
        self.assertEqual(20, resolved.get("priority"))
        self.assertEqual("test_cluster", resolved.get("cluster_id"))

    def test_runopts_resolve_missing_required(self) -> None:
        opts = self.get_runopts()

        cfg = {
            "priority": 20,
            "cluster_id": "test_cluster",
        }

        with self.assertRaises(InvalidRunConfigException):
            opts.resolve(cfg)

    def test_runopts_resolve_bad_type(self) -> None:
        opts = self.get_runopts()

        cfg = {
            "run_as": "foobar",
            "cluster_id": 123,
        }

        with self.assertRaises(InvalidRunConfigException):
            opts.resolve(cfg)

    def test_runopts_resolve_unioned(self) -> None:
        # runconfigs is a union of all run opts for all schedulers
        # make sure  opts resolves run configs that have more
        # configs than it knows about
        opts = self.get_runopts()
        cfg = {
            "run_as": "foobar",
            "some_other_opt": "baz",
        }

        resolved = opts.resolve(cfg)
        self.assertEqual("foobar", resolved.get("run_as"))
        self.assertEqual(10, resolved.get("priority"))
        self.assertIsNone(resolved.get("cluster_id"))
        self.assertEqual("baz", resolved.get("some_other_opt"))

    def test_runopts_is_type(self) -> None:
        # primitive types
        self.assertTrue(runopts.is_type(3, int))
        self.assertFalse(runopts.is_type("foo", int))
        # List[str]
        self.assertFalse(runopts.is_type(None, List[str]))
        self.assertTrue(runopts.is_type([], List[str]))
        self.assertTrue(runopts.is_type(["a", "b"], List[str]))

    def test_runopts_iter(self) -> None:
        runopts = self.get_runopts()
        for name, opt in runopts:
            self.assertEqual(opt, runopts.get(name))


class GetTypeNameTest(unittest.TestCase):
    def test_get_type_name(self) -> None:
        self.assertEqual("int", get_type_name(int))
        self.assertEqual("list", get_type_name(list))
        self.assertEqual("typing.Union[str, int]", get_type_name(Union[str, int]))
        self.assertEqual("typing.List[int]", get_type_name(List[int]))
        self.assertEqual("typing.Dict[str, int]", get_type_name(Dict[str, int]))
        self.assertEqual(
            "typing.List[typing.List[int]]", get_type_name(List[List[int]])
        )


class MacrosTest(unittest.TestCase):
    def test_substitute(self) -> None:
        v = macros.Values(
            img_root="img_root",
            app_id="app_id",
            replica_id="replica_id",
            base_img_root="base_img_root",
        )
        for key, val in asdict(v).items():
            template = f"tmpl-{getattr(macros, key)}"
            self.assertEqual(v.substitute(template), f"tmpl-{val}")

    def test_apply(self) -> None:
        role = Role(
            name="test",
            image="test_image",
            entrypoint="foo.py",
            args=[macros.img_root],
            env={"FOO": macros.app_id},
        )
        v = macros.Values(
            img_root="img_root",
            app_id="app_id",
            replica_id="replica_id",
            base_img_root="base_img_root",
        )
        newrole = v.apply(role)
        self.assertNotEqual(newrole, role)
        self.assertEqual(newrole.args, ["img_root"])
        self.assertEqual(newrole.env, {"FOO": "app_id"})


def get_dummy_application(role: str) -> AppDef:
    trainer = Role(
        role,
        "test_image",
        entrypoint="main_script.py",
        args=["--train"],
        num_replicas=2,
    )
    return AppDef(name="test_app", roles=[trainer])


def test_empty_fn() -> AppDef:
    """Empty function that returns dummy app"""
    return get_dummy_application("trainer")


def test_fn_with_bool(flag: bool = False) -> AppDef:
    """Dummy app with or without flag

    Args:
        flag: flag param
    """
    if flag:
        return get_dummy_application("trainer-with-flag")
    else:
        return get_dummy_application("trainer-without-flag")


def test_fn_with_bool_opional(flag: Optional[bool] = None) -> AppDef:
    """Dummy app with or without flag

    Args:
        flag: flag param
    """
    if flag:
        return get_dummy_application("trainer-with-flag")
    else:
        return get_dummy_application("trainer-without-flag")


def test_empty_fn_no_docstring() -> AppDef:
    return get_dummy_application("trainer")


def _test_complex_fn(
    app_name: str,
    containers: List[str],
    roles_scripts: Dict[str, str],
    num_cpus: Optional[List[int]] = None,
    num_gpus: Optional[Dict[str, int]] = None,
    nnodes: int = 4,
    first_arg: Optional[str] = None,
    *roles_args: str,
) -> AppDef:
    """Creates complex application, testing all possible complex types

    Args:
        app_name: AppDef name
        containers: List of containers
        roles_scripts: Dict role_name -> role_script
    """
    num_roles = len(roles_scripts)
    if not num_cpus:
        num_cpus = [1] * num_roles
    if not num_gpus:
        num_gpus = {}
        for role in roles_scripts.keys():
            num_gpus[role] = 1
    roles = []
    for idx, (role_name, role_script) in enumerate(roles_scripts.items()):
        container_img = containers[idx]
        cpus = num_cpus[idx]
        gpus = num_gpus[role_name]
        if first_arg:
            args = [first_arg, *roles_args]
        else:
            args = [*roles_args]
        role = Role(
            role_name,
            image=container_img,
            entrypoint=role_script,
            args=args,
            resource=Resource(cpu=cpus, gpu=gpus, memMB=1),
            num_replicas=nnodes,
        )
        roles.append(role)
    return AppDef(app_name, roles)


_TEST_VAR_ARGS: Optional[Tuple[object, ...]] = None


def _test_var_args(foo: str, *args: str, bar: str = "asdf") -> AppDef:
    """
    test component for mixing var args with kwargs.
    Args:
        foo: arg
        args: varargs
        bar: kwarg
    """
    global _TEST_VAR_ARGS
    _TEST_VAR_ARGS = (foo, args, bar)
    return AppDef(name="varargs")


_TEST_VAR_ARGS_FIRST: Optional[Tuple[object, ...]] = None


def _test_var_args_first(*args: str, bar: str = "asdf") -> AppDef:
    """
    test component for mixing var args with kwargs.
    Args:
        args: varargs
        bar: kwarg
    """
    global _TEST_VAR_ARGS_FIRST
    _TEST_VAR_ARGS_FIRST = (args, bar)
    return AppDef(name="varargs")


_TEST_SINGLE_LETTER: Optional[str] = None


def _test_single_letter(c: str) -> AppDef:
    global _TEST_SINGLE_LETTER
    _TEST_SINGLE_LETTER = c
    return AppDef(name="varargs")


class AppDefLoadTest(unittest.TestCase):
    def assert_apps(self, expected_app: AppDef, actual_app: AppDef) -> None:
        self.assertDictEqual(asdict(expected_app), asdict(actual_app))

    def _get_role_args(self) -> List[str]:
        return ["--train", "data_source", "random", "--epochs", "128"]

    def _get_expected_app_with_default(self) -> AppDef:
        role_args = self._get_role_args()
        return _test_complex_fn(
            "test_app",
            ["img1", "img2"],
            {"worker": "worker.py", "master": "master.py"},
            None,
            None,
            4,
            None,
            *role_args,
        )

    def _get_args_with_default(self) -> List[str]:
        role_args = self._get_role_args()
        return [
            "--app_name",
            "test_app",
            "--containers",
            "img1,img2",
            "--roles_scripts",
            "worker=worker.py,master=master.py",
            "--",
            *role_args,
        ]

    def _get_expected_app_with_all_args(self) -> AppDef:
        role_args = self._get_role_args()
        return _test_complex_fn(
            "test_app",
            ["img1", "img2"],
            {"worker": "worker.py", "master": "master.py"},
            [1, 2],
            {"worker": 1, "master": 4},
            8,
            "first_arg",
            *role_args,
        )

    def _get_app_args(self) -> List[str]:
        role_args = self._get_role_args()
        return [
            "--app_name",
            "test_app",
            "--containers",
            "img1,img2",
            "--roles_scripts",
            "worker=worker.py,master=master.py",
            "--num_cpus",
            "1,2",
            "--num_gpus",
            "worker=1,master=4",
            "--nnodes",
            "8",
            "--first_arg",
            "first_arg",
            "--",
            *role_args,
        ]

    def test_load_from_fn_empty(self) -> None:
        actual_app = from_function(test_empty_fn, [])
        expected_app = get_dummy_application("trainer")
        self.assert_apps(expected_app, actual_app)

    def test_load_from_fn_complex_all_args(self) -> None:
        expected_app = self._get_expected_app_with_all_args()
        app_args = self._get_app_args()
        actual_app = from_function(_test_complex_fn, app_args)
        self.assert_apps(expected_app, actual_app)

    def test_required_args(self) -> None:
        with patch.object(sys, "exit") as exit_mock:
            try:
                from_function(_test_complex_fn, [])
            except Exception:
                # ignore any errors, since function should fail
                pass
        exit_mock.assert_called_once()

    def test_load_from_fn_with_default(self) -> None:
        expected_app = self._get_expected_app_with_default()
        app_args = self._get_args_with_default()
        actual_app = from_function(_test_complex_fn, app_args)
        self.assert_apps(expected_app, actual_app)

    def test_varargs(self) -> None:
        from_function(
            _test_var_args,
            [
                "--foo",
                "fooval",
                "--bar",
                "barval",
                "arg1",
                "arg2",
            ],
        )
        self.assertEqual(_TEST_VAR_ARGS, ("fooval", ("arg1", "arg2"), "barval"))

    def test_bool_true(self) -> None:
        app_def = from_function(
            test_fn_with_bool,
            [
                "--flag",
                "True",
            ],
        )
        self.assertEqual("trainer-with-flag", app_def.roles[0].name)
        app_def = from_function(
            test_fn_with_bool,
            [
                "--flag",
                "true",
            ],
        )
        self.assertEqual("trainer-with-flag", app_def.roles[0].name)

    def test_bool_false(self) -> None:
        app_def = from_function(
            test_fn_with_bool,
            [
                "--flag",
                "False",
            ],
        )
        self.assertEqual("trainer-without-flag", app_def.roles[0].name)
        app_def = from_function(
            test_fn_with_bool,
            [
                "--flag",
                "false",
            ],
        )
        self.assertEqual("trainer-without-flag", app_def.roles[0].name)

    def test_bool_none(self) -> None:
        app_def = from_function(
            test_fn_with_bool,
            [],
        )
        self.assertEqual("trainer-without-flag", app_def.roles[0].name)

    def test_varargs_only_flag_first(self) -> None:
        from_function(
            _test_var_args_first,
            [
                "--",
                "--foo",
                "fooval",
                "barval",
                "arg1",
                "arg2",
            ],
        )
        self.assertEqual(
            _TEST_VAR_ARGS_FIRST,
            (("--foo", "fooval", "barval", "arg1", "arg2"), "asdf"),
        )

    def test_varargs_only_arg_first(self) -> None:
        from_function(
            _test_var_args_first,
            [
                "fooval",
                "--foo",
                "barval",
                "arg1",
                "arg2",
            ],
        )
        self.assertEqual(
            _TEST_VAR_ARGS_FIRST,
            (("fooval", "--foo", "barval", "arg1", "arg2"), "asdf"),
        )

    def test_single_letter(self) -> None:
        from_function(
            _test_single_letter,
            [
                "-c",
                "arg1",
            ],
        )
        self.assertEqual(
            _TEST_SINGLE_LETTER,
            "arg1",
        )

        from_function(
            _test_single_letter,
            [
                "--c",
                "arg2",
            ],
        )
        self.assertEqual(
            _TEST_SINGLE_LETTER,
            "arg2",
        )

    # pyre-ignore[3]
    def _get_argument_help(
        self, parser: argparse.ArgumentParser, name: str
    ) -> Optional[Tuple[str, Any]]:
        actions = parser._actions
        for action in actions:
            if action.dest == name:
                return action.help or "", action.default
        return None

    def test_argparster_complex_fn_partial(self) -> None:
        parser = _create_args_parser(_test_complex_fn)
        self.assertTupleEqual(
            ("AppDef name", None),
            none_throws(self._get_argument_help(parser, "app_name")),
        )
        self.assertTupleEqual(
            ("List of containers", None),
            none_throws(self._get_argument_help(parser, "containers")),
        )
        self.assertTupleEqual(
            ("Dict role_name -> role_script", None),
            none_throws(self._get_argument_help(parser, "roles_scripts")),
        )
        self.assertTupleEqual(
            (" ", None), none_throws(self._get_argument_help(parser, "num_cpus"))
        )
        self.assertTupleEqual(
            (" ", None), none_throws(self._get_argument_help(parser, "num_gpus"))
        )
        self.assertTupleEqual(
            (" ", 4), none_throws(self._get_argument_help(parser, "nnodes"))
        )
        self.assertTupleEqual(
            (" ", None), none_throws(self._get_argument_help(parser, "first_arg"))
        )
        self.assertTupleEqual(
            (" ", None), none_throws(self._get_argument_help(parser, "roles_args"))
        )
