# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from shutil import copy2, rmtree
from tempfile import mkdtemp
from typing import Any, List, Mapping, Optional, Set, Type

import ray  # @manual # noqa: F401
# from ray.dashboard.modules.job.common import JobStatus
from ray._private.job_manager import JobStatus
from ray.autoscaler import sdk as ray_autoscaler_sdk
from ray.dashboard.modules.job.sdk import JobSubmissionClient
from torchx.schedulers.api import (
    AppDryRunInfo,
    AppState,
    DescribeAppResponse,
    Scheduler,
    Stream,
)
from torchx.schedulers.ids import make_unique
from torchx.schedulers.ray.ray_common import RayActor
from torchx.specs import AppDef, CfgVal, SchedulerBackend, macros, runopts

_has_ray = True


_logger: logging.Logger = logging.getLogger(__name__)

ray_status_to_torchx_appstate = {
    JobStatus.PENDING: AppState.PENDING,
    JobStatus.RUNNING: AppState.RUNNING,
    JobStatus.SUCCEEDED: AppState.SUCCEEDED,
    JobStatus.FAILED: AppState.FAILED,
    JobStatus.STOPPED: AppState.CANCELLED,
}


class _EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def _serialize(
    actors: List[RayActor], dirpath: str, output_filename: str = "actors.json"
) -> None:
    actors_json = json.dumps(actors, cls=_EnhancedJSONEncoder)
    with open(os.path.join(dirpath, output_filename), "w") as tmp:
        json.dump(actors_json, tmp)


def has_ray() -> bool:
    """Indicates whether Ray is installed in the current Python environment."""
    return _has_ray


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
        cluster_address:
            The existing cluster IP address to connect to
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
    """

    app_id: str
    cluster_config_file: Optional[str] = None
    cluster_name: Optional[str] = None
    cluster_address: Optional[str] = None
    copy_scripts: bool = False
    copy_script_dirs: bool = False
    scripts: Set[str] = field(default_factory=set)
    actors: List[RayActor] = field(default_factory=list)


class RayScheduler(Scheduler):
    client: JobSubmissionClient

    def __init__(self, session_name: str) -> None:
        super().__init__("ray", session_name)

    # TODO: Add address as a potential CLI argument after writing ray.status() or passing in config file
    def run_opts(self) -> runopts:
        opts = runopts()
        opts.add(
            "cluster_config_file",
            type_=str,
            required=False,
            help="Use CLUSTER_CONFIG_FILE to access or create the Ray cluster.",
        )
        opts.add(
            "cluster_name",
            type_=str,
            help="Override the configured cluster name.",
        )
        opts.add(
            "cluster_address",
            type_=str,
            required=False,
            default="127.0.0.1",
            help="Use ray status to get the cluster_address",
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
        return opts

    def schedule(self, dryrun_info: AppDryRunInfo[RayJob]) -> str:
        cfg: RayJob = dryrun_info.request

        # Create serialized actors for ray_driver.py
        actors = cfg.actors

        dirpath = mkdtemp()

        _serialize(actors, dirpath)

        ip_address = cfg.cluster_address
        if cfg.cluster_config_file:
            ip_address = ray_autoscaler_sdk.get_head_node_ip(cfg.cluster_config_file)

        dashboard_port = 8265

        # Create Job Client
        self.client = JobSubmissionClient(f"http://{ip_address}:{dashboard_port}")

        # entrypoint = os.path.join("ray", "ray_driver.py")

        # 1. Copy scripts and directories
        if cfg.scripts:
            for script in cfg.scripts:
                if cfg.copy_script_dirs:
                    script_dir = os.path.dirname(script)
                    copy2(script_dir, dirpath)
                else:
                    copy2(script, dirpath)

        # 2. Copy Ray driver utilities
        current_directory = os.path.dirname(os.path.abspath(__file__))
        copy2(os.path.join(current_directory, "ray", "ray_driver.py"), dirpath)
        copy2(os.path.join(current_directory, "ray", "ray_common.py"), dirpath)
        print("DIR PATH")
        job_id: str = self.client.submit_job(
            # we will pack, hash, zip, upload, register working_dir in GCS of ray cluster
            # and use it to configure your job execution.
            entrypoint=f"python ray_driver.py",
            runtime_env={"working_dir": dirpath},
            # job_id = cfg.app_id
        )

        rmtree(dirpath)

        return job_id

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
        self.client.stop_job(app_id)

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        status = self.client.get_job_status(app_id)
        status = ray_status_to_torchx_appstate[status]
        return DescribeAppResponse(app_id=app_id, state=status)

    def log_iter(
        self,
        app_id: str,
        role_name: Optional[str] = None,
        k: int = 0,
        regex: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        should_tail: bool = False,
        streams: Optional[Stream] = None,
    ) -> List[str]:
        # TODO: support regex, tailing, streams etc..
        logs = self.client.get_job_logs(app_id)
        return logs


def create_scheduler(session_name: str, **kwargs: Any) -> RayScheduler:
    if not has_ray():
        raise RuntimeError("Ray is not installed in the current Python environment.")

    return RayScheduler(session_name=session_name)
