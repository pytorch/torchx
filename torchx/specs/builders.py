# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import argparse
import inspect
import os
from argparse import Namespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from torchx.specs.api import BindMount, MountType, VolumeMount
from torchx.specs.file_linter import get_fn_docstring, TorchXArgumentHelpFormatter
from torchx.util.types import decode, decode_optional, get_argparse_param_type, is_bool

from .api import AppDef, DeviceMount


def _create_args_parser(
    cmpnt_fn: Callable[..., AppDef],
    cmpnt_defaults: Optional[Dict[str, str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> argparse.ArgumentParser:
    parameters = inspect.signature(cmpnt_fn).parameters
    return _create_args_parser_from_parameters(
        cmpnt_fn, parameters, cmpnt_defaults, config
    )


def _create_args_parser_from_parameters(
    cmpnt_fn: Callable[..., Any],  # pyre-ignore[2]
    parameters: Mapping[str, inspect.Parameter],
    cmpnt_defaults: Optional[Dict[str, str]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> argparse.ArgumentParser:
    function_desc, args_desc = get_fn_docstring(cmpnt_fn)
    script_parser = argparse.ArgumentParser(
        prog=f"torchx run <run args...> {cmpnt_fn.__name__} ",
        description=function_desc,
        formatter_class=TorchXArgumentHelpFormatter,
        # enables components to have "h" as a parameter
        # otherwise argparse by default adds -h/--help as the help argument
        # we still add --help but reserve "-"h" to be used as a component argument
        add_help=False,
    )
    # add help manually since we disabled auto help to allow "h" in component arg
    script_parser.add_argument(
        "--help",
        action="help",
        default=argparse.SUPPRESS,
        help="show this help message and exit",
    )

    class _reminder_action(argparse.Action):
        def __call__(
            self,
            parser: argparse.ArgumentParser,
            namespace: argparse.Namespace,
            values: Any,
            option_string: Optional[str] = None,
        ) -> None:
            setattr(
                namespace,
                self.dest,
                (self.default or "").split() if len(values) == 0 else values,
            )

    for param_name, parameter in parameters.items():
        param_desc = args_desc[parameter.name]
        args: Dict[str, Any] = {
            "help": param_desc,
            "type": get_argparse_param_type(parameter),
        }
        # set defaults specified in the component function declaration
        if parameter.default != inspect.Parameter.empty:
            if is_bool(type(parameter.default)):
                args["default"] = str(parameter.default)
            else:
                args["default"] = parameter.default

        # set defaults supplied directly to this method (overwrites the declared defaults)
        # the defaults are given as str (as option values passed from CLI) since
        # these are typically read from .torchxconfig
        if cmpnt_defaults and param_name in cmpnt_defaults:
            args["default"] = cmpnt_defaults[param_name]

        if parameter.kind == inspect._ParameterKind.VAR_POSITIONAL:
            args["nargs"] = argparse.REMAINDER
            args["action"] = _reminder_action
            script_parser.add_argument(param_name, **args)
        else:
            arg_names = [f"--{param_name}"]
            if len(param_name) == 1:
                arg_names = [f"-{param_name}"] + arg_names
            if "default" not in args:
                if (config and param_name not in config) or not config:
                    args["required"] = True

            script_parser.add_argument(*arg_names, **args)
    return script_parser


def _merge_config_values_with_args(
    parsed_args: argparse.Namespace, config: Dict[str, Any]
) -> None:
    for key, val in config.items():
        if key in parsed_args:
            setattr(parsed_args, key, val)


def parse_args(
    cmpnt_fn: Callable[..., Any],  # pyre-ignore[2]
    cmpnt_args: List[str],
    cmpnt_defaults: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Namespace:
    """
    Parse passed arguments, defaults, and config values into a namespace for
    a component function.

    Args:
    cmpnt_fn: Component function
    cmpnt_args: Function args
    cmpnt_defaults: Additional default values for parameters of ``app_fn``
                        (overrides the defaults set on the fn declaration)
    config: Optional dict containing additional configuration for the component from a passed config file

    Returns:
    A Namespace object with the args, defaults, and config values incorporated.
    """

    script_parser = _create_args_parser(cmpnt_fn, cmpnt_defaults, config)
    parsed_args = script_parser.parse_args(cmpnt_args)
    if config:
        _merge_config_values_with_args(parsed_args, config)

    return parsed_args


def materialize_appdef(
    cmpnt_fn: Callable[..., Any],  # pyre-ignore[2]
    cmpnt_args: List[str],
    cmpnt_defaults: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> AppDef:
    """
    Creates an application by running user defined ``app_fn``.

    ``app_fn`` has the following restrictions:
        * Name must be ``app_fn``
        * All arguments should be annotated
        * Supported argument types:
            - primitive: int, str, float
            - Dict[primitive, primitive]
            - List[primitive]
            - Optional[Dict[primitive, primitive]]
            - Optional[List[primitive]]
        * ``app_fn`` can define a vararg (*arg) at the end
        * There should be a docstring for the function that defines
            All arguments in a google-style format
        * There can be default values for the function arguments.
        * The return object must be ``AppDef``

    Args:
        cmpnt_fn: Component function
        cmpnt_args: Function args
        cmpnt_defaults: Additional default values for parameters of ``app_fn``
                          (overrides the defaults set on the fn declaration)
        config: Optional dict containing additional configuration for the component from a passed config file
    Returns:
        An application spec
    """

    function_args = []
    var_arg = []
    kwargs = {}

    parsed_args = parse_args(cmpnt_fn, cmpnt_args, cmpnt_defaults, config)

    parameters = inspect.signature(cmpnt_fn).parameters
    for param_name, parameter in parameters.items():
        arg_value = getattr(parsed_args, param_name)
        parameter_type = parameter.annotation
        parameter_type = decode_optional(parameter_type)
        arg_value = decode(arg_value, parameter_type)
        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            var_arg = arg_value
        elif parameter.kind == inspect.Parameter.KEYWORD_ONLY:
            kwargs[param_name] = arg_value
        elif parameter.kind == inspect.Parameter.VAR_KEYWORD:
            raise TypeError("**kwargs are not supported for component definitions")
        else:
            function_args.append(arg_value)
    if len(var_arg) > 0 and var_arg[0] == "--":
        var_arg = var_arg[1:]

    appdef = cmpnt_fn(*function_args, *var_arg, **kwargs)
    if not isinstance(appdef, AppDef):
        raise TypeError(
            f"Expected a component that returns `AppDef`, but got `{type(appdef)}`"
        )

    return appdef


def make_app_handle(scheduler_backend: str, session_name: str, app_id: str) -> str:
    return f"{scheduler_backend}://{session_name}/{app_id}"


_MOUNT_OPT_MAP: Mapping[str, str] = {
    "type": "type",
    "destination": "dst",
    "dst": "dst",
    "target": "dst",
    "read_only": "readonly",
    "readonly": "readonly",
    "source": "src",
    "src": "src",
    "perm": "perm",
}


def parse_mounts(opts: List[str]) -> List[Union[BindMount, VolumeMount, DeviceMount]]:
    """
    parse_mounts parses a list of options into typed mounts following a similar
    format to Dockers bind mount.

    Multiple mounts can be specified in the same list. ``type`` must be
    specified first in each.

    Ex:
        type=bind,src=/host,dst=/container,readonly,[type=bind,src=...,dst=...]

    Supported types:
        BindMount: type=bind,src=<host path>,dst=<container path>[,readonly]
        VolumeMount: type=volume,src=<name/id>,dst=<container path>[,readonly]
        DeviceMount: type=device,src=/dev/<dev>[,dst=<container path>][,perm=rwm]
    """
    mount_opts = []
    cur = {}
    for opt in opts:
        key, _, val = opt.partition("=")
        if key not in _MOUNT_OPT_MAP:
            raise KeyError(
                f"unknown mount option {key}, must be one of {list(_MOUNT_OPT_MAP.keys())}"
            )
        key = _MOUNT_OPT_MAP[key]
        if key == "type":
            cur = {}
            mount_opts.append(cur)
        elif len(mount_opts) == 0:
            raise KeyError("type must be specified first")
        cur[key] = val

    mounts = []
    for opts in mount_opts:
        typ = opts.get("type")
        if typ == MountType.BIND:
            src_path = opts["src"]
            if src_path.startswith("~"):
                src_path = os.path.expanduser(src_path)
            mounts.append(
                BindMount(
                    src_path=src_path,
                    dst_path=opts["dst"],
                    read_only="readonly" in opts,
                )
            )
        elif typ == MountType.VOLUME:
            mounts.append(
                VolumeMount(
                    src=opts["src"], dst_path=opts["dst"], read_only="readonly" in opts
                )
            )
        elif typ == MountType.DEVICE:
            src = opts["src"]
            dst = opts.get("dst", src)
            perm = opts.get("perm", "rwm")
            for c in perm:
                if c not in "rwm":
                    raise ValueError(
                        f"{c} is not a valid permission flags must one of r,w,m"
                    )
            mounts.append(DeviceMount(src_path=src, dst_path=dst, permissions=perm))
        else:
            valid = list(str(item.value) for item in MountType)
            raise ValueError(f"invalid mount type {repr(typ)}, must be one of {valid}")
    return mounts
