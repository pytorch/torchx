# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from shutil import copy2, rmtree
from typing import Any, cast, Dict, Final, Iterable, List, Optional, Tuple  # noqa

import urllib3

from torchx.schedulers.api import (
    AppDryRunInfo,
    AppState,
    DescribeAppResponse,
    filter_regex,
    ListAppResponse,
    Scheduler,
    split_lines,
    Stream,
)
from torchx.schedulers.ids import make_unique
from torchx.schedulers.ray.ray_common import RayActor, TORCHX_RANK0_HOST
from torchx.specs import AppDef, macros, NONE, ReplicaStatus, Role, RoleStatus, runopts
from torchx.workspace.dir_workspace import TmpDirWorkspaceMixin
from typing_extensions import TypedDict


try:
    from ray.autoscaler import sdk as ray_autoscaler_sdk
    from ray.dashboard.modules.job.common import JobStatus
    from ray.dashboard.modules.job.sdk import JobSubmissionClient

    _has_ray = True

except ImportError:
    _has_ray = False


def has_ray() -> bool:
    """Indicates whether Ray is installed in the current Python environment."""
    return _has_ray


class RayOpts(TypedDict, total=False):
    cluster_config_file: Optional[str]
    cluster_name: Optional[str]
    dashboard_address: Optional[str]
    working_dir: Optional[str]
    requirements: Optional[str]


if _has_ray:
    _logger: logging.Logger = logging.getLogger(__name__)

    _ray_status_to_torchx_appstate: Dict[JobStatus, AppState] = {
        JobStatus.PENDING: AppState.PENDING,
        JobStatus.RUNNING: AppState.RUNNING,
        JobStatus.SUCCEEDED: AppState.SUCCEEDED,
        JobStatus.FAILED: AppState.FAILED,
        JobStatus.STOPPED: AppState.CANCELLED,
    }

    class _EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, o: RayActor):  # pyre-ignore[3]
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)

    def serialize(
        actors: List[RayActor], dirpath: str, output_filename: str = "actors.json"
    ) -> None:
        actors_json = json.dumps(actors, cls=_EnhancedJSONEncoder)
        with open(os.path.join(dirpath, output_filename), "w") as tmp:
            json.dump(actors_json, tmp)

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
            dashboard_address:
                The existing dashboard IP address to connect to
            working_dir:
                The working directory to copy to the cluster
            requirements:
                The libraries to install on the cluster per requirements.txt
            actors:
                The Ray actors which represent the job to be run. This attribute is
                dumped to a JSON file and copied to the cluster where `ray_main.py`
                uses it to initiate the job.
        """

        app_id: str
        working_dir: str
        cluster_config_file: Optional[str] = None
        cluster_name: Optional[str] = None
        dashboard_address: Optional[str] = None
        requirements: Optional[str] = None
        actors: List[RayActor] = field(default_factory=list)

    class RayScheduler(TmpDirWorkspaceMixin, Scheduler[RayOpts]):
        """
        RayScheduler is a TorchX scheduling interface to Ray. The job def
        workers will be launched as Ray actors

        The job environment is specified by the TorchX workspace. Any files in
        the workspace will be present in the Ray job unless specified in
        ``.torchxignore``. Python dependencies will be read from the
        ``requirements.txt`` file located at the root of the workspace unless
        it's overridden via ``-c ...,requirements=foo/requirements.txt``.

        **Config Options**

        .. runopts::
            class: torchx.schedulers.ray_scheduler.create_scheduler

        **Compatibility**

        .. compatibility::
            type: scheduler
            features:
                cancel: true
                logs: |
                    Partial support. Ray only supports a single log stream so
                    only a dummy "ray/0" combined log role is supported.
                    Tailing and time seeking are not supported.
                distributed: true
                describe: |
                    Partial support. RayScheduler will return job status but
                    does not provide the complete original AppSpec.
                workspaces: true
                mounts: false
                elasticity: Partial support. Multi role jobs are not supported.

        """

        def __init__(
            self, session_name: str, ray_client: Optional[JobSubmissionClient] = None
        ) -> None:
            # NOTE: make sure any new init options are supported in create_scheduler(...)
            super().__init__("ray", session_name)

            # w/o Final None check in _get_ray_client does not work as it pyre assumes mutability
            self._ray_client: Final[Optional[JobSubmissionClient]] = ray_client

        def _get_ray_client(
            self, job_submission_netloc: Optional[str] = None
        ) -> JobSubmissionClient:
            if self._ray_client is not None:
                client_netloc = urllib3.util.parse_url(
                    self._ray_client.get_address()
                ).netloc
                if job_submission_netloc and job_submission_netloc != client_netloc:
                    raise ValueError(
                        f"client netloc ({client_netloc}) does not match job netloc ({job_submission_netloc})"
                    )
                return self._ray_client
            elif os.getenv("RAY_ADDRESS"):
                return JobSubmissionClient(os.getenv("RAY_ADDRESS"))
            elif not job_submission_netloc:
                raise Exception(
                    "RAY_ADDRESS env variable or a scheduler with an attached Ray JobSubmissionClient is expected."
                    " See https://docs.ray.io/en/latest/cluster/jobs-package-ref.html#job-submission-sdk for more info"
                )
            return JobSubmissionClient(f"http://{job_submission_netloc}")

        # TODO: Add address as a potential CLI argument after writing ray.status() or passing in config file
        def _run_opts(self) -> runopts:
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
                "dashboard_address",
                type_=str,
                required=False,
                default="127.0.0.1:8265",
                help="Use ray status to get the dashboard address you will submit jobs against",
            )
            opts.add("requirements", type_=str, help="Path to requirements.txt")
            return opts

        def schedule(self, dryrun_info: AppDryRunInfo[RayJob]) -> str:
            cfg: RayJob = dryrun_info.request

            # Create serialized actors for ray_driver.py
            actors = cfg.actors
            dirpath = cfg.working_dir
            serialize(actors, dirpath)

            job_submission_addr: str = ""
            if cfg.cluster_config_file:
                job_submission_addr = ray_autoscaler_sdk.get_head_node_ip(
                    cfg.cluster_config_file
                )  # pragma: no cover
            elif cfg.dashboard_address:
                job_submission_addr = cfg.dashboard_address
            else:
                raise RuntimeError(
                    "Either `dashboard_address` or `cluster_config_file` must be specified"
                )

            # 0. Create Job Client
            client = self._get_ray_client(job_submission_netloc=job_submission_addr)

            # 1. Copy Ray driver utilities
            current_directory = os.path.dirname(os.path.abspath(__file__))
            copy2(os.path.join(current_directory, "ray", "ray_driver.py"), dirpath)
            copy2(os.path.join(current_directory, "ray", "ray_common.py"), dirpath)
            runtime_env = {"working_dir": dirpath}
            if cfg.requirements:
                runtime_env["pip"] = cfg.requirements

            # 1. Submit Job via the Ray Job Submission API
            try:
                job_id: str = client.submit_job(
                    submission_id=cfg.app_id,
                    # we will pack, hash, zip, upload, register working_dir in GCS of ray cluster
                    # and use it to configure your job execution.
                    entrypoint="python3 ray_driver.py",
                    runtime_env=runtime_env,
                )

            finally:
                if dirpath.startswith(tempfile.gettempdir()):
                    rmtree(dirpath)

            # Encode job submission client in job_id
            return f"{job_submission_addr}-{job_id}"

        def _submit_dryrun(self, app: AppDef, cfg: RayOpts) -> AppDryRunInfo[RayJob]:
            app_id = make_unique(app.name)

            working_dir = app.roles[0].image
            if not os.path.exists(working_dir):
                raise RuntimeError(
                    f"Role image must be a valid directory, got: {working_dir} "
                )

            requirements: Optional[str] = cfg.get("requirements")
            if requirements is None:
                workspace_reqs = os.path.join(working_dir, "requirements.txt")
                if os.path.exists(workspace_reqs):
                    requirements = workspace_reqs

            cluster_cfg = cfg.get("cluster_config_file")
            if cluster_cfg:
                if not isinstance(cluster_cfg, str) or not os.path.isfile(cluster_cfg):
                    raise ValueError(
                        "The cluster configuration file must be a YAML file."
                    )

                job: RayJob = RayJob(
                    app_id,
                    cluster_config_file=cluster_cfg,
                    requirements=requirements,
                    working_dir=working_dir,
                )

            else:  # pragma: no cover
                dashboard_address = cfg.get("dashboard_address")
                job: RayJob = RayJob(
                    app_id=app_id,
                    dashboard_address=dashboard_address,
                    requirements=requirements,
                    working_dir=working_dir,
                )
            job.cluster_name = cfg.get("cluster_name")

            for role in app.roles:
                for replica_id in range(role.num_replicas):
                    # Replace the ${img_root}, ${app_id}, and ${replica_id} placeholders
                    # in arguments and environment variables.
                    replica_role = macros.Values(
                        img_root=role.image,
                        app_id=app_id,
                        replica_id=str(replica_id),
                        rank0_env=TORCHX_RANK0_HOST,
                    ).apply(role)

                    actor = RayActor(
                        name=role.name,
                        min_replicas=role.min_replicas,
                        command=[replica_role.entrypoint] + replica_role.args,
                        env=replica_role.env,
                        num_cpus=max(1, replica_role.resource.cpu),
                        num_gpus=max(0, replica_role.resource.gpu),
                    )

                    job.actors.append(actor)

            if len(app.roles) > 1 and app.roles[0].min_replicas is not None:
                raise ValueError("min_replicas is only supported with single role jobs")

            return AppDryRunInfo(job, repr)

        def _validate(self, app: AppDef, scheduler: str) -> None:
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

        def wait_until_finish(self, app_id: str, timeout: int = 30) -> None:
            """
            ``wait_until_finish`` waits until the specified job has finished
            with a given timeout. This is intended for testing. Programmatic
            usage should use the runner wait method instead.
            """

            start = time.time()
            while time.time() - start <= timeout:
                status_info = self._get_job_status(app_id)
                status = status_info
                if status in {JobStatus.SUCCEEDED, JobStatus.STOPPED, JobStatus.FAILED}:
                    break
                time.sleep(1)

        def _parse_app_id(self, app_id: str) -> Tuple[str, str]:
            # find index of '-' in the first :\d+-
            m = re.search(r":\d+-", app_id)
            if m:
                sep = m.span()[1]
                addr = app_id[: sep - 1]
                app_id = app_id[sep:]
                return addr, app_id

            addr, _, app_id = app_id.partition("-")
            return addr, app_id

        def _cancel_existing(self, app_id: str) -> None:  # pragma: no cover
            addr, app_id = self._parse_app_id(app_id)
            client = self._get_ray_client(job_submission_netloc=addr)
            client.stop_job(app_id)

        def _get_job_status(self, app_id: str) -> JobStatus:
            addr, app_id = self._parse_app_id(app_id)
            client = self._get_ray_client(job_submission_netloc=addr)
            status = client.get_job_status(app_id)
            if isinstance(status, str):
                return cast(JobStatus, status)
            return status.status

        def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
            job_status_info = self._get_job_status(app_id)
            state = _ray_status_to_torchx_appstate[job_status_info]
            roles = [Role(name="ray", num_replicas=1, image="<N/A>")]

            # get ip_address and put it in hostname

            roles_statuses = [
                RoleStatus(
                    role="ray",
                    replicas=[
                        ReplicaStatus(
                            id=0,
                            role="ray",
                            hostname=NONE,
                            state=state,
                        )
                    ],
                )
            ]
            return DescribeAppResponse(
                app_id=app_id,
                state=state,
                msg=job_status_info,
                roles_statuses=roles_statuses,
                roles=roles,
            )

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
        ) -> Iterable[str]:
            # TODO: support tailing, streams etc..
            addr, app_id = self._parse_app_id(app_id)
            client: JobSubmissionClient = self._get_ray_client(
                job_submission_netloc=addr
            )
            logs: str = client.get_job_logs(app_id)
            iterator = split_lines(logs)
            if regex:
                return filter_regex(regex, iterator)
            return iterator

        def list(self) -> List[ListAppResponse]:
            client = self._get_ray_client()
            jobs = client.list_jobs()
            netloc = urllib3.util.parse_url(client.get_address()).netloc
            return [
                ListAppResponse(
                    app_id=f"{netloc}-{details.submission_id}",
                    state=_ray_status_to_torchx_appstate[details.status],
                )
                for details in jobs
            ]


def create_scheduler(
    session_name: str, ray_client: Optional[JobSubmissionClient] = None, **kwargs: Any
) -> "RayScheduler":
    if not has_ray():  # pragma: no cover
        raise ModuleNotFoundError(
            "Ray is not installed in the current Python environment."
        )

    return RayScheduler(session_name=session_name, ray_client=ray_client)
