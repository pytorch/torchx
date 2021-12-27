# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Mapping, Any, Dict, Iterable, List, Optional, Set, Type

from torchx.schedulers.api import AppDryRunInfo, DescribeAppResponse, Scheduler, Stream
from torchx.schedulers.ids import make_unique
from torchx.specs import AppDef, CfgVal, SchedulerBackend, macros, runopts


try:
    import ray  # @manual # noqa: F401

    _has_ray = True
except ImportError:
    _has_ray = False


_logger: logging.Logger = logging.getLogger(__name__)


def has_ray() -> bool:
    """Indicates whether Ray is installed in the current Python environment."""
    return _has_ray


@dataclass
class RayActor:
    """Describes an actor (a.k.a. role in TorchX terms).

    Attributes:
        name:
            The name of the actor.
        command:
            The command that the actor should run as a subprocess.
        env:
            The environment variables to set before executing the command.
        num_replicas:
            The number of replicas (i.e. Ray actors) to run.
        num_cpus:
            The number of CPUs to allocate.
        num_gpus:
            The number of GPUs to allocate.
    """

    name: str
    command: str
    env: Dict[str, str] = field(default_factory=dict)
    num_replicas: int = 1
    num_cpus: int = 1
    num_gpus: int = 0
    # TODO: memory_size, max_retries, retry_policy


@dataclass
class RayJob:
    """Represents a job that should be run on a Ray cluster.

    Attributes:
        app_id:
            The unique ID of the application (a.k.a. job).
        cluster_config_file:
            The Ray cluster configuration file.
        cluster_name:
            The cluster name to use.
        copy_script:
            A boolean value indicating whether to copy the script files to the
            cluster.
        copy_script_dir:
            A boolean value indicating whether to copy the directories
            containing the scripts to the cluster.
        scripts:
            The set of scripts to copy to the cluster.
        actors:
            The Ray actors which represent the job to be run. This attribute is
            dumped to a JSON file and copied to the cluster where `ray_main.py`
            uses it to initiate the job.
        verbose:
            A boolean value indicating whether to enable verbose output.
    """

    app_id: str
    cluster_config_file: str
    cluster_name: Optional[str] = None
    copy_scripts: bool = False
    copy_script_dirs: bool = False
    scripts: Set[str] = field(default_factory=set)
    actors: List[RayActor] = field(default_factory=list)
    verbose: bool = False


class RayScheduler(Scheduler):
    def __init__(self, session_name: str) -> None:
        super().__init__("ray", session_name)

    def run_opts(self) -> runopts:
        opts = runopts()
        opts.add(
            "cluster_config_file",
            type_=str,
            required=True,
            help="Use CLUSTER_CONFIG_FILE to access or create the Ray cluster.",
        )
        opts.add(
            "cluster_name",
            type_=str,
            help="Override the configured cluster name.",
        )
        opts.add(
            "copy_scripts",
            type_=bool,
            default=False,
            help="Copy the Python script(s) to the cluster.",
        )
        opts.add(
            "copy_script_dirs",
            type_=bool,
            default=False,
            help="Copy the directories containing the Python scripts to the cluster.",
        )
        opts.add(
            "verbose",
            type_=bool,
            default=False,
            help="Enable verbose output.",
        )
        return opts

    def schedule(self, dryrun_info: AppDryRunInfo[RayJob]) -> str:
        raise NotImplementedError()

    def _submit_dryrun(
        self, app: AppDef, cfg: Mapping[str, CfgVal]
    ) -> AppDryRunInfo[RayJob]:
        app_id = make_unique(app.name)

        cluster_cfg = cfg.get("cluster_config_file")
        if not isinstance(cluster_cfg, str) or not os.path.isfile(cluster_cfg):
            raise ValueError("The cluster configuration file must be a YAML file.")

        job: RayJob = RayJob(app_id, cluster_cfg)

        # pyre-ignore[24]: Generic type `type` expects 1 type parameter
        def set_job_attr(cfg_name: str, cfg_type: Type) -> None:
            cfg_value = cfg.get(cfg_name)
            if cfg_value is None:
                return

            if not isinstance(cfg_value, cfg_type):
                raise TypeError(
                    f"The configuration value '{cfg_name}' must be of type {cfg_type.__name__}."
                )

            setattr(job, cfg_name, cfg_value)

        set_job_attr("cluster_name", str)
        set_job_attr("copy_scripts", bool)
        set_job_attr("copy_script_dirs", bool)
        set_job_attr("verbose", bool)

        for role in app.roles:
            # Replace the ${img_root}, ${app_id}, and ${replica_id} placeholders
            # in arguments and environment variables.
            role = macros.Values(
                img_root=role.image, app_id=app_id, replica_id="${rank}"
            ).apply(role)

            actor = RayActor(
                name=role.name,
                command=" ".join([role.entrypoint] + role.args),
                env=role.env,
                num_replicas=max(1, role.num_replicas),
                num_cpus=max(1, role.resource.cpu),
                num_gpus=max(0, role.resource.gpu),
            )

            job.actors.append(actor)

            if job.copy_scripts or job.copy_script_dirs:
                # Find out the actual user script.
                for arg in role.args:
                    if arg.endswith(".py"):
                        job.scripts.add(arg)

        return AppDryRunInfo(job, repr)

    def _validate(self, app: AppDef, scheduler: SchedulerBackend) -> None:
        if scheduler != "ray":
            raise ValueError(
                f"An unknown scheduler backend '{scheduler}' has been passed to the Ray scheduler."
            )

        if app.metadata:
            _logger.warning("The Ray scheduler does not use metadata information.")

        for role in app.roles:
            if role.resource.capabilities:
                _logger.warning(
                    "The Ray scheduler does not support custom resource capabilities."
                )
                break

        for role in app.roles:
            if role.port_map:
                _logger.warning("The Ray scheduler does not support port mapping.")
                break

    def _cancel_existing(self, app_id: str) -> None:
        raise NotImplementedError()

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        raise NotImplementedError()

    def log_iter(
        self,
        app_id: str,
        role_name: str,
        k: int = 0,
        regex: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        should_tail: bool = False,
        streams: Optional[Stream] = None,
    ) -> Iterable[str]:
        raise NotImplementedError()


def create_scheduler(session_name: str, **kwargs: Any) -> RayScheduler:
    if not has_ray():
        raise RuntimeError("Ray is not installed in the current Python environment.")

    return RayScheduler(session_name=session_name)
