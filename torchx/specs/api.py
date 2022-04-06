#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import copy
import inspect
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from string import Template
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

import yaml
from torchx.specs.file_linter import TorchXArgumentHelpFormatter, get_fn_docstring
from torchx.util.types import (
    decode_from_string,
    decode_optional,
    get_argparse_param_type,
    is_bool,
    is_primitive,
    to_dict,
)


# ========================================
# ==== Distributed AppDef API =======
# ========================================
@dataclass
class Resource:
    """
    Represents resource requirements for a ``Role``.

    Args:
        cpu: number of logical cpu cores. The definition of a CPU core depends
            on the scheduler. See your scheduler documentation for how a logical
            CPU core maps to physical cores and threads.
        gpu: number of gpus
        memMB: MB of ram
        capabilities: additional hardware specs (interpreted by scheduler)

    Note: you should prefer to use named_resources instead of specifying the raw
    resource requirement directly.
    """

    cpu: int
    gpu: int
    memMB: int
    capabilities: Dict[str, Any] = field(default_factory=dict)

    @staticmethod
    def copy(original: "Resource", **capabilities: Any) -> "Resource":
        """
        Copies a resource and applies new capabilities. If the same capabilities
        are present in the original resource and as parameter, the one from parameter
        will be used.
        """
        res_capabilities = dict(original.capabilities)
        res_capabilities.update(capabilities)
        return Resource(
            cpu=original.cpu,
            gpu=original.gpu,
            memMB=original.memMB,
            capabilities=res_capabilities,
        )


# sentinel value used for cases when resource does not matter (e.g. ignored)
NULL_RESOURCE: Resource = Resource(cpu=-1, gpu=-1, memMB=-1)

# used as "*" scheduler backend
ALL: str = "all"

# sentinel value used to represent missing string attributes, such as image or entrypoint
MISSING: str = "<MISSING>"

# sentinel value used to represent "unset" optional string attributes
NONE: str = "<NONE>"


class macros:
    """
    Defines macros that can be used in the elements of ``Role.args``
    values of ``Role.env``. The macros will be substituted at runtime
    to their actual values.

    .. warning:: Macros used fields of :py:class:`Role` other than the ones
                 mentioned above, are NOT substituted.

    Available macros:

    1. ``img_root`` - root directory of the pulled container.image
    2. ``app_id`` - application id as assigned by the scheduler
    3. ``replica_id`` - unique id for each instance of a replica of a Role,
                        for instance a role with 3 replicas could have the 0, 1, 2
                        as replica ids. Note that when the container fails and is
                        replaced, the new container will have the same ``replica_id``
                        as the one it is replacing. For instance if node 1 failed and
                        was replaced by the scheduler the replacing node will also
                        have ``replica_id=1``.

    Example:

    ::

     # runs: hello_world.py --app_id ${app_id}
     trainer = Role(
                name="trainer",
                entrypoint="hello_world.py",
                args=["--app_id", macros.app_id],
                env={"IMAGE_ROOT_DIR": macros.img_root})
     app = AppDef("train_app", roles=[trainer])
     app_handle = session.run(app, scheduler="local_docker", cfg={})

    """

    img_root = "${img_root}"
    base_img_root = "${base_img_root}"
    app_id = "${app_id}"
    replica_id = "${replica_id}"

    # rank0_env will be filled with the name of the environment variable that
    # provides the master host address. To get the actual hostname the
    # environment variable must be resolved by the app via either shell
    # expansion (wrap sh/bash) or via the application.
    # This may not be available on all schedulers.
    rank0_env = "${rank0_env}"

    @dataclass
    class Values:
        img_root: str
        app_id: str
        replica_id: str
        rank0_env: str
        base_img_root: str = "DEPRECATED"

        def apply(self, role: "Role") -> "Role":
            """
            apply applies the values to a copy the specified role and returns it.
            """

            role = copy.deepcopy(role)
            role.args = [self.substitute(arg) for arg in role.args]
            role.env = {key: self.substitute(arg) for key, arg in role.env.items()}
            return role

        def substitute(self, arg: str) -> str:
            """
            substitute applies the values to the template arg.
            """
            return Template(arg).safe_substitute(**asdict(self))


class RetryPolicy(str, Enum):
    """
    Defines the retry policy for the ``Roles`` in the ``AppDef``.
    The policy defines the behavior when the role replica encounters a failure:

    1. unsuccessful (non zero) exit code
    2. hardware/host crashes
    3. preemption
    4. eviction

    .. note:: Not all retry policies are supported by all schedulers.
              However all schedulers must support ``RetryPolicy.APPLICATION``.
              Please refer to the scheduler's documentation for more information
              on the retry policies they support and behavior caveats (if any).

    1. REPLICA: Replaces the replica instance. Surviving replicas are untouched.
                Use with ``torch_dist_role`` to have torch coordinate restarts
                and membership changes. Otherwise, it is up to the application to
                deal with failed replica departures and replacement replica admittance.
    2. APPLICATION: Restarts the entire application.

    """

    REPLICA = "REPLICA"
    APPLICATION = "APPLICATION"


class MountType(str, Enum):
    BIND = "bind"
    VOLUME = "volume"
    DEVICE = "device"


@dataclass
class BindMount:
    """
    Defines a bind mount to `mount --bind` a host path into the worker
    environment. See scheduler documentation on how bind mounts operate for each
    scheduler.

    Args:
        src_path: the path on the host
        dst_path: the path in the worker environment/container
        read_only: whether the mount should be read only
    """

    src_path: str
    dst_path: str
    read_only: bool = False


@dataclass
class VolumeMount:
    """
    Defines a persistent volume mount to mount into the worker environment.
    Args:
       src: the name or ID of the volume to mount
       dst_path: the path in the worker environment/container
       read_only: whether the mount should be read only
    """

    src: str
    dst_path: str
    read_only: bool = False


@dataclass
class DeviceMount:
    """
    Defines a host device to mount into the container.
    Args:
       src_path: the path on the host
       dst_path: the path in the worker environment/container
       permissions: the permissions to set on the device. Default: read, write, mknode
    """

    src_path: str
    dst_path: str
    permissions: str = "rwm"


@dataclass
class Role:
    """
    A set of nodes that perform a specific duty within the ``AppDef``.
    Examples:

    1. Distributed data parallel app - made up of a single role (trainer).

    2. App with parameter server - made up of multiple roles (trainer, ps).

    .. note:: An ``image`` is a software bundle that is installed on the container
              scheduled by the scheduler. The container on the scheduler dictates
              what an image actually is. An image could be as simple as a tar-ball
              or map to a docker image. The scheduler typically knows how to "pull"
              the image given an image name (str), which could be a simple name
              (e.g. docker image) or a url e.g. ``s3://path/my_image.tar``).

    Usage:

    ::

     trainer = Role(name="trainer",
                    image = "pytorch/torch:1",
                    entrypoint = "my_trainer.py"
                    args = ["--arg", "foo", ENV_VAR="FOOBAR"],
                    num_replicas = 4,
                    resource = Resource(cpu=1, gpu=1, memMB=500),
                    port_map={"tcp_store":8080, "tensorboard": 8081},
                    metadata={"local_cwd.property", value})

    Args:
            name: name of the role
            image: a software bundle that is installed on a container.
            entrypoint: command (within the container) to invoke the role
            args: commandline arguments to the entrypoint cmd
            env: environment variable mappings
            replicas: number of container replicas to run
            max_retries: max number of retries before giving up
            retry_policy: retry behavior upon replica failures
            resource: Resource requirement for the role. The role should be scheduled
                by the scheduler on ``num_replicas`` container, each of them should have at
                least ``resource`` guarantees.
            port_map: Port mapping for the role. The key is the unique identifier of the port
                e.g. "tensorboard": 9090
            metadata: Free form information that is associated with the role, for example
                scheduler specific data. The key should follow the pattern: ``$scheduler.$key``
            mounts: a list of mounts on the machine
    """

    name: str
    image: str
    base_image: Optional[str] = None  # DEPRECATED DO NOT SET, WILL BE REMOVED SOON
    entrypoint: str = MISSING
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    num_replicas: int = 1
    max_retries: int = 0
    retry_policy: RetryPolicy = RetryPolicy.APPLICATION
    resource: Resource = NULL_RESOURCE
    port_map: Dict[str, int] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    mounts: List[Union[BindMount, VolumeMount, DeviceMount]] = field(
        default_factory=list
    )

    def pre_proc(
        self,
        scheduler: str,
        # pyre-fixme[24]: AppDryRunInfo was designed to work with Any request object
        dryrun_info: "AppDryRunInfo",
        # pyre-fixme[24]: AppDryRunInfo was designed to work with Any request object
    ) -> "AppDryRunInfo":
        """
        Modifies the scheduler request based on the role specific configuration.
        The method is invoked for each role during scheduler ``submit_dryrun``.
        If there are multiple roles, the method is invoked for each role in
        order that is defined by the ``AppDef.roles`` list.
        """
        return dryrun_info


@dataclass
class AppDef:
    """
    Represents a distributed application made up of multiple ``Roles``
    and metadata. Contains the necessary information for the driver
    to submit this app to the scheduler.

    Args:
        name: Name of application
        roles: List of roles
        metadata: metadata to the app (treament of metadata is scheduler dependent)
    """

    name: str
    roles: List[Role] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)


class AppState(int, Enum):
    """
    State of the application. An application starts from an initial
    ``UNSUBMITTED`` state and moves through ``SUBMITTED``, ``PENDING``,
    ``RUNNING`` states finally reaching a terminal state:
    ``SUCCEEDED``,``FAILED``, ``CANCELLED``.

    If the scheduler supports preemption, the app moves from a ``RUNNING``
    state to ``PENDING`` upon preemption.

    If the user stops the application, then the application state moves
    to ``STOPPED``, then to ``CANCELLED`` when the job is actually cancelled
    by the scheduler.

    1. UNSUBMITTED - app has not been submitted to the scheduler yet
    2. SUBMITTED - app has been successfully submitted to the scheduler
    3. PENDING - app has been submitted to the scheduler pending allocation
    4. RUNNING - app is running
    5. SUCCEEDED - app has successfully completed
    6. FAILED - app has unsuccessfully completed
    7. CANCELLED - app was cancelled before completing
    8. UNKNOWN - app state is unknown
    """

    UNSUBMITTED = 0
    SUBMITTED = 1
    PENDING = 2
    RUNNING = 3
    SUCCEEDED = 4
    FAILED = 5
    CANCELLED = 6
    UNKNOWN = 7

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return f"{self.name} ({self.value})"


_TERMINAL_STATES: List[AppState] = [
    AppState.SUCCEEDED,
    AppState.FAILED,
    AppState.CANCELLED,
]

_STARTED_STATES: List[AppState] = _TERMINAL_STATES + [
    AppState.RUNNING,
]


def is_terminal(state: AppState) -> bool:
    return state in _TERMINAL_STATES


def is_started(state: AppState) -> bool:
    return state in _STARTED_STATES


# =======================
# ==== Status API =======
# =======================

# replica and app share the same states, simply alias it for now
ReplicaState = AppState


@dataclass
class ReplicaStatus:
    """
    The status of the replica during the job execution.

    Args:
        id: The node rank, note: this is not a worker rank.
        state: The current state of the node.
        role: The role name
        hostname: The hostname where the replica is running
        structured_error_msg: Error message if any, None if job succeeded.
    """

    id: int
    state: ReplicaState
    role: str
    hostname: str
    structured_error_msg: str = NONE


@dataclass
class RoleStatus:
    """
    The status of the role during the job execution.

    Args:
        role: Role name
        replicas: List of replica statuses
    """

    role: str
    replicas: List[ReplicaStatus]


@dataclass
class AppStatus:
    """
    The runtime status of the ``AppDef``. The scheduler can
    return an arbitrary text message (msg field).
    If any error occurs, scheduler can populate ``structured_error_msg``
    with json response.

    ``replicas`` represent the statuses of the replicas in the job. If the job
    runs with multiple retries, the parameter will contain the statuses of the
    most recent retry. Note: if the previous retries failed, but the most recent
    retry succeeded or in progress, ``replicas`` will not contain occurred errors.
    """

    state: AppState
    num_restarts: int = 0
    msg: str = ""
    structured_error_msg: str = NONE
    ui_url: Optional[str] = None
    roles: List[RoleStatus] = field(default_factory=list)

    def is_terminal(self) -> bool:
        return is_terminal(self.state)

    def __repr__(self) -> str:
        app_status_dict = asdict(self)
        structured_error_msg = app_status_dict.pop("structured_error_msg")
        if structured_error_msg != NONE:
            structured_error_msg_parsed = json.loads(structured_error_msg)
        else:
            structured_error_msg_parsed = NONE
        app_status_dict["structured_error_msg"] = structured_error_msg_parsed
        app_status_dict["state"] = repr(app_status_dict["state"])
        return yaml.dump({"AppStatus": app_status_dict})


# valid run cfg values; only support primitives (str, int, float, bool, List[str])
# TODO(wilsonhong): python 3.9+ supports list[T] in typing, which can be used directly
# in isinstance(). Should replace with that.
# see: https://docs.python.org/3/library/stdtypes.html#generic-alias-type
CfgVal = Union[str, int, float, bool, List[str], None]


T = TypeVar("T")


class AppDryRunInfo(Generic[T]):
    """
    Returned by ``Scheduler.submit_dryrun``. Represents the
    request that would have been made to the scheduler.
    The ``fmt_str()`` method of this object should return a
    pretty formatted string representation of the underlying
    request object such that ``print(info)`` yields a human
    readable representation of the underlying request.
    """

    def __init__(self, request: T, fmt: Callable[[T], str]) -> None:
        self.request = request
        self._fmt = fmt

        # fields below are only meant to be used by
        # Scheduler or Session implementations
        # and are back references to the parameters
        # to dryrun() that returned this AppDryRunInfo object
        # thus they are set in Session.dryrun() and Scheduler.submit_dryrun()
        # manually rather than through constructor arguments
        # DO NOT create getters or make these public
        # unless there is a good reason to
        self._app: Optional[AppDef] = None
        self._cfg: Mapping[str, CfgVal] = {}
        self._scheduler: Optional[str] = None

    def __repr__(self) -> str:
        return self._fmt(self.request)


def get_type_name(tp: Type[CfgVal]) -> str:
    """
    Gets the type's name as a string. If ``tp` is a primitive class like int, str, etc, then
    uses its attribute ``__name__``. Otherwise, use ``str(tp)``.

    Note: we use this method to print out generic typing like List[str].
    """
    if tp.__module__ != "typing" and hasattr(tp, "__name__"):
        return tp.__name__
    else:
        return str(tp)


@dataclass
class runopt:
    """
    Represents the metadata about the specific run option
    """

    default: CfgVal
    opt_type: Type[CfgVal]
    is_required: bool
    help: str


class runopts:
    """
    Holds the accepted scheduler run configuration
    keys, default value (if any), and help message string.
    These options are provided by the ``Scheduler`` and validated
    in ``Session.run`` against user provided run cfg.
    Allows ``None`` default values. Required opts must NOT have a
    non-None default.

    .. important:: This class has no accessors because it is intended to
                   be constructed and returned by ``Scheduler.run_config_options``
                   and printed out as a "help" tool or as part of an exception msg.

    Usage:

    .. code-block:: python

     opts = runopts()

     opts.add("run_as_user", type_=str, help="user to run the job as")
     opts.add("cluster_id", type_=int, help="cluster to submit the job", required=True)
     opts.add("priority", type_=float, default=0.5, help="job priority")
     opts.add("preemptible", type_=bool, default=False, help="is the job preemptible")

     # invalid
     opts.add("illegal", default=10, required=True)
     opts.add("bad_type", type=str, default=10)

     opts.check(cfg)
     print(opts)

    """

    def __init__(self) -> None:
        self._opts: Dict[str, runopt] = {}

    def __iter__(self) -> Iterator[Tuple[str, runopt]]:
        return self._opts.items().__iter__()

    def __len__(self) -> int:
        return len(self._opts)

    @staticmethod
    def is_type(obj: CfgVal, tp: Type[CfgVal]) -> bool:
        """
        Returns True if ``obj`` is type of ``tp``. Similar to isinstance() but supports
        tp = List[str], thus can be used to validate ConfigValue.
        """
        try:
            return isinstance(obj, tp)
        except TypeError:
            if isinstance(obj, list):
                return all(isinstance(e, str) for e in obj)
            else:
                return False

    def get(self, name: str) -> Optional[runopt]:
        """
        Returns option if any was registered, or None otherwise
        """
        return self._opts.get(name, None)

    def resolve(self, cfg: Mapping[str, CfgVal]) -> Dict[str, CfgVal]:
        """
        Checks the given config against this ``runopts`` and sets default configs
        if not set.

        .. note:: Extra configs unknown to this run option are ignored.

        """

        resolved_cfg: Dict[str, CfgVal] = {**cfg}

        for cfg_key, runopt in self._opts.items():
            val = resolved_cfg.get(cfg_key)

            # check required opt
            if runopt.is_required and val is None:
                raise InvalidRunConfigException(
                    f"Required run option: {cfg_key}, must be provided and not `None`",
                    cfg_key,
                    cfg,
                )

            # check type (None matches all types)
            if val is not None and not runopts.is_type(val, runopt.opt_type):
                raise InvalidRunConfigException(
                    f"Run option: {cfg_key}, must be of type: {get_type_name(runopt.opt_type)},"
                    f" but was: {val} ({type(val).__name__})",
                    cfg_key,
                    cfg,
                )

            # not required and not set, set to default
            if val is None:
                resolved_cfg[cfg_key] = runopt.default
        return resolved_cfg

    def cfg_from_str(self, cfg_str: str) -> Dict[str, CfgVal]:
        """
        Parses scheduler ``runcfg`` from a string literal and returns
        a cfg map where the cfg values have been cast into the appropriate
        types as specified by this runopts object. Unknown keys are ignored
        and not returned in the resulting map.

        .. note:: Unlike the method ``resolve``, this method does NOT resolve
                  default options or check that the required options are actually
                  present in the given ``cfg_str``. This method is intended to be
                  called before calling ``resolve()`` when the input is a string
                  encoded run cfg. That is to fully resolve the cfg, call
                  ``opt.resolve(opt.cfg_from_str(cfg_literal))``.

        If the ``cfg_str`` is an empty string, then an empty
        ``cfg`` is returned. Otherwise, at least one kv-pair delimited by
        ``"="`` (equal) is expected.

        Either ``","`` (comma) or ``";"`` (semi-colon)
        can be used to delimit multiple kv-pairs.

        ``CfgVal`` allows ``List`` of primitives, which can be passed as
        either ``","`` or ``";"`` (semi-colon) delimited. Since the same
        delimiters are used to delimit between cfg kv pairs, this method
        interprets the last (trailing) ``","`` or ``";"`` as the delimiter between
        kv pairs. See example below.



        Examples:

        .. doctest::

         opts = runopts()
         opts.add("FOO", type_=List[str], default=["a"], help="an optional list option")
         opts.add("BAR", type_=str, required=True, help="a required str option")

         # required and default options not checked
         # method returns strictly parsed cfg from the cfg literal string
         opts.cfg_from_str("") == {}

         # however, unknown options are ignored
         # since the value type is unknown hence cannot cast to the correct type
         opts.cfg_from_str("UNKNOWN=VALUE") == {}

         opts.cfg_from_str("FOO=v1") == {"FOO": "v1"}

         opts.cfg_from_str("FOO=v1,v2") == {"FOO": ["v1", "v2"]}
         opts.cfg_from_str("FOO=v1;v2") == {"FOO": ["v1", "v2"]}

         opts.cfg_from_str("FOO=v1,v2,BAR=v3") == {"FOO": ["v1", "v2"], "BAR": "v3"}
         opts.cfg_from_str("FOO=v1;v2,BAR=v3") == {"FOO": ["v1", "v2"], "BAR": "v3"}
         opts.cfg_from_str("FOO=v1;v2;BAR=v3") == {"FOO": ["v1", "v2"], "BAR": "v3"}

        """

        def _cast_to_type(value: str, opt_type: Type[CfgVal]) -> CfgVal:
            if opt_type == bool:
                return value.lower() == "true"
            elif opt_type == List[str]:
                # lists may be ; or , delimited
                # also deal with trailing "," by removing empty strings
                return [v for v in value.replace(";", ",").split(",") if v]
            else:
                # pyre-ignore[19]
                return opt_type(value)

        cfg: Dict[str, CfgVal] = {}
        for key, val in to_dict(cfg_str).items():
            runopt_ = self.get(key)
            if runopt_:
                cfg[key] = _cast_to_type(val, runopt_.opt_type)
        return cfg

    def add(
        self,
        cfg_key: str,
        type_: Type[CfgVal],
        help: str,
        default: CfgVal = None,
        required: bool = False,
    ) -> None:
        """
        Adds the ``config`` option with the given help string and ``default``
        value (if any). If the ``default`` is not specified then this option
        is a required option.
        """
        if required and default is not None:
            raise ValueError(
                f"Required option: {cfg_key} must not specify default value. Given: {default}"
            )
        if default is not None:
            if not runopts.is_type(default, type_):
                raise TypeError(
                    f"Option: {cfg_key}, must be of type: {type_}."
                    f" Given: {default} ({type(default).__name__})"
                )

        self._opts[cfg_key] = runopt(default, type_, required, help)

    def __repr__(self) -> str:
        required = [(key, opt) for key, opt in self._opts.items() if opt.is_required]
        optional = [
            (key, opt) for key, opt in self._opts.items() if not opt.is_required
        ]

        out = "    usage:\n        "
        for i, (key, opt) in enumerate(required + optional):
            contents = f"{key}={key.upper()}"
            if not opt.is_required:
                contents = f"[{contents}]"
            if i > 0:
                contents = "," + contents
            out += contents

        sections = [("required", required), ("optional", optional)]

        for section, opts in sections:
            if len(opts) == 0:
                continue
            out += f"\n\n    {section} arguments:"
            for key, opt in opts:
                default = "" if opt.is_required else f", {opt.default}"
                out += f"\n        {key}={key.upper()} ({get_type_name(opt.opt_type)}{default})"
                out += f"\n            {opt.help}"

        return out


class InvalidRunConfigException(Exception):
    """
    Raised when the supplied run cfg does not satisfy the
    ``runopts``, either due to missing required configs or value
    type mismatch.
    """

    def __init__(
        self, invalid_reason: str, cfg_key: str, cfg: Mapping[str, CfgVal]
    ) -> None:
        given = str(cfg) if cfg else "<EMPTY>"
        super().__init__(f"{invalid_reason}. Given: {given}")
        self.cfg_key = cfg_key


class MalformedAppHandleException(Exception):
    """
    Raised when APIs are given a bad app handle.
    """

    def __init__(self, app_handle: str) -> None:
        super().__init__(
            f"{app_handle} is not of the form: <scheduler_backend>://<session_name>/<app_id>"
        )


class UnknownSchedulerException(Exception):
    def __init__(self, scheduler_backend: str) -> None:
        super().__init__(
            f"Scheduler backend: {scheduler_backend} does not exist."
            f" Use session.scheduler_backends() to see all supported schedulers"
        )


# encodes information about a running app in url format
# {scheduler_backend}://{session_name}/{app_id}
AppHandle = str


class UnknownAppException(Exception):
    """
    Raised by ``Session`` APIs when either the application does not
    exist or the application is not owned by the session.
    """

    def __init__(self, app_handle: "AppHandle") -> None:
        super().__init__(
            f"Unknown app = {app_handle}. Did you forget to call session.run()?"
            f" Otherwise, the app may have already finished and purged by the scheduler"
        )


def make_app_handle(scheduler_backend: str, session_name: str, app_id: str) -> str:
    return f"{scheduler_backend}://{session_name}/{app_id}"


def parse_app_handle(app_handle: AppHandle) -> Tuple[str, str, str]:
    """
    parses the app handle into ```(scheduler_backend, session_name, and app_id)```
    """

    # parse it manually b/c currently torchx does not
    # define allowed characters nor length for session name and app_id
    import re

    pattern = r"(?P<scheduler_backend>.+)://(?P<session_name>.+)/(?P<app_id>.+)"
    match = re.match(pattern, app_handle)
    if not match:
        raise MalformedAppHandleException(app_handle)
    gd = match.groupdict()
    return gd["scheduler_backend"], gd["session_name"], gd["app_id"]


def _create_args_parser(
    cmpnt_fn: Callable[..., AppDef], cmpnt_defaults: Optional[Dict[str, str]] = None
) -> argparse.ArgumentParser:
    parameters = inspect.signature(cmpnt_fn).parameters
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
            script_parser.add_argument(param_name, **args)
        else:
            arg_names = [f"--{param_name}"]
            if len(param_name) == 1:
                arg_names = [f"-{param_name}"] + arg_names
            if "default" not in args:
                args["required"] = True
            script_parser.add_argument(*arg_names, **args)
    return script_parser


# FIXME this function invokes the component fn to materialize the AppDef
#      rename to something more appropriate like - materialize_appdef or eval_component_fn
#      seee: https://github.com/pytorch/torchx/issues/368
def from_function(
    cmpnt_fn: Callable[..., AppDef],
    cmpnt_args: List[str],
    cmpnt_defaults: Optional[Dict[str, str]] = None,
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
    Returns:
        An application spec
    """

    script_parser = _create_args_parser(cmpnt_fn, cmpnt_defaults)
    parsed_args = script_parser.parse_args(cmpnt_args)

    function_args = []
    var_arg = []
    kwargs = {}

    parameters = inspect.signature(cmpnt_fn).parameters
    for param_name, parameter in parameters.items():
        arg_value = getattr(parsed_args, param_name)
        parameter_type = parameter.annotation
        parameter_type = decode_optional(parameter_type)
        if is_bool(parameter_type):
            arg_value = arg_value.lower() == "true"
        elif not is_primitive(parameter_type):
            arg_value = decode_from_string(arg_value, parameter_type)
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

    return cmpnt_fn(*function_args, *var_arg, **kwargs)


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
            mounts.append(
                BindMount(
                    src_path=opts["src"],
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
