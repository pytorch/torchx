# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Set, Type

import ray

from torchx.schedulers.api import AppDryRunInfo, DescribeAppResponse, Scheduler
from torchx.schedulers.ids import make_unique
from torchx.specs.api import AppDef, RunConfig, SchedulerBackend, macros, runopts


logger: logging.Logger = logging.getLogger(__name__)


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
        memory_size:
            The amount of memory, in MBs, to allocate.
    """

    name: str
    command: str
    env: Dict[str, str] = field(default_factory=dict)
    ray_env = field(default_factory=dict)
    num_replicas: int = 1
    num_cpus: int = 1
    num_gpus: int = 1
    memory_size: int = 1
    max_retries: int = 5
    # TODO: retry_policy

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
        start_cluster:
            A boolean value indicating whether to start the cluster if needed.
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
    start_cluster: bool = False
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
            "start_cluster",
            type_=bool,
            default=False,
            help="Start the cluster if needed.",
        )
        opts.add(
            "copy_scripts",
            type_=bool,
            default=False,
            help="Copy the script(s) to the cluster. Supported only for PyTorch DDP jobs.",
        )
        opts.add(
            "copy_script_dirs",
            type_=bool,
            default=False,
            help="Copy the directories containing the scripts to the cluster. "
            "Supported only for PyTorch DDP jobs.",
        )
        opts.add(
            "verbose",
            type_=bool,
            default=False,
            help="Enable verbose output.",
        )
        return opts

    def schedule(self, dryrun_info: AppDryRunInfo[RayJob]) -> str:
        cfg: RayJob = dryrun_info.request

        # 1. Start the cluster using `ray up` if requested by `cfg.start_cluster`.
        #
        # 2. Copy `cfg.scripts` or their directories to `~/<cfg.app_id>` on the
        #    cluster using `ray rsync-up` if requested by `cfg.copy_scripts` or
        #    `cfg.copy_script_dirs`.
        #
        # 3. Dump `cfg.actors` to a JSON file (e.g. `/tmp/<cfg.app_id>-actors.json`)
        #    and copy to `~/<cfg.app_id>` on the cluster using `ray rsync-up`.
        #
        # 4. Copy `ray_main.py` which handles the actual job execution to
        #    `~/<cfg.app_id>` on the cluster using `ray rsync-up`.
        #
        # 5. Start `ray_main.py` as a background process on the head node using
        #    `ray exec`.

        return cfg.app_id

    def _submit_dryrun(self, app: AppDef, cfg: RunConfig) -> AppDryRunInfo[RayJob]:
        app_id = make_unique(app.name)

        cluster_cfg = cfg.get("cluster_config_file")
        if not isinstance(cluster_cfg, str) or not os.path.isfile(cluster_cfg):
            raise ValueError("The cluster configuration file must be a YAML file.")

        job = RayJob(app_id, cluster_cfg)

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
        set_job_attr("start_cluster", bool)
        set_job_attr("copy_scripts", bool)
        set_job_attr("copy_script_dirs", bool)
        set_job_attr("verbose", bool)

        warn_unused_copy_option = True

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
                num_replicas=role.num_replicas,
                num_cpus=role.resource.cpu,
                num_gpus=role.resource.gpu,
                memory_size=role.resource.memMB,
            )

            job.actors.append(actor)

            if job.copy_scripts or job.copy_script_dirs:
                # We support copying scripts to the cluster only for DDP jobs.
                if (
                    # fmt: off
                    actor.command.find(" -m torch.distributed.run ") > 1 or
                    actor.command.find(" -m torch.distributed.launch ") > 1 or
                    actor.command.startswith("torchrun")
                    # fmt: on
                ):
                    # Find out the actual user script.
                    for arg in role.args:
                        if arg.endswith(".py"):
                            job.scripts.add(arg)
                            break
                else:
                    if warn_unused_copy_option:
                        logger.warning(
                            "The `copy_scripts` and `copy_script_dirs` options are valid "
                            "only for PyTorch DDP jobs."
                        )

                        warn_unused_copy_option = False

        return AppDryRunInfo(job, repr)

    def _validate(self, app: AppDef, scheduler: SchedulerBackend) -> None:
        if scheduler != "ray":
            raise ValueError(
                f"An unknown scheduler backend '{scheduler}' has been passed to the Ray scheduler."
            )

        if app.metadata:
            logger.warning("The Ray scheduler does not use metadata information.")

        for role in app.roles:
            if role.resource.capabilities:
                logger.warning(
                    "The Ray scheduler does not support custom resource capabilities."
                )
                break

        for role in app.roles:
            if role.port_map:
                logger.warning("The Ray scheduler does not support port mapping.")
                break

    def _cancel_existing(self, app_id: str) -> None:
        # By using `ray exec` run a simple shell or Python script that:
        #
        # 1. Makes sure that `/var/run/torchx-ray/<app_id>.status` does not
        #    exist.
        #
        # 2. Retrieves the pid of `ray_main.py` from `/var/run/torchx-ray/<app_id>.pid`.
        #
        # 3. Sends a SIGTERM to the process.
        pass

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        # By using `ray exec` run a simple shell or Python script that:
        #
        #  1. Checks if `/var/run/torchx-ray/<app_id>.status` exists which
        #     should contain the exit status of the `ray_main.py` process. If
        #     exists, return SUCCEEDED (0), CANCELED (1), or FAILED (2) based on
        #     the status.
        #
        #  2. Checks if `/var/run/torchx-ray/<app_id>.pid` exists which contains
        #     the pid of `ray_main.py` process. If not, marks the job as
        #     'UNSUBMITTED'.
        #
        #  3. Checks if there is a running process with the retrieved pid. If
        #     not, marks the job as 'FAILED'.
        #
        #  4. In order to avoid race conditions, before returning the value from
        #     steps 2 or 3, repeats step 1 and ensures that status file does not
        #     exist.

        return None

    def log_iter(
        self,
        app_id: str,
        role_name: str,
        k: int = 0,
        regex: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        should_tail: bool = False,
    ) -> Iterable[str]:
        # TBD
        pass


def create_scheduler(session_name: str, **kwargs: Any) -> RayScheduler:
    return RayScheduler(session_name=session_name)
