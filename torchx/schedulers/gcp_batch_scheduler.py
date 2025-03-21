#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""

This contains the TorchX GCP Batch scheduler which can be used to run TorchX
components directly on GCP Batch.

This scheduler is in prototype stage and may change without notice.

Prerequisites
==============

You need to have a GCP project configured to use Batch by enabling and setting it up.
See https://cloud.google.com/batch/docs/get-started#prerequisites

"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

import torchx
import yaml

from torchx.schedulers.api import (
    AppDryRunInfo,
    DescribeAppResponse,
    ListAppResponse,
    Scheduler,
    Stream,
)
from torchx.schedulers.ids import make_unique
from torchx.specs.api import AppDef, AppState, macros, Resource, Role, runopts
from torchx.util.strings import normalize_str
from typing_extensions import TypedDict


if TYPE_CHECKING:
    from google.cloud import batch_v1


JOB_STATE: Dict[str, AppState] = {
    "STATE_UNSPECIFIED": AppState.UNKNOWN,
    "QUEUED": AppState.SUBMITTED,
    "SCHEDULED": AppState.PENDING,
    "RUNNING": AppState.RUNNING,
    "SUCCEEDED": AppState.SUCCEEDED,
    "FAILED": AppState.FAILED,
    "DELETION_IN_PROGRESS": AppState.UNKNOWN,
}

GPU_COUNT_TO_TYPE: Dict[int, str] = {
    1: "a2-highgpu-1g",
    2: "a2-highgpu-2g",
    4: "a2-highgpu-4g",
    8: "a2-highgpu-8g",
    16: "a2-highgpu-16g",
}

GPU_TYPE_TO_COUNT: Dict[str, int] = {v: k for k, v in GPU_COUNT_TO_TYPE.items()}

LABEL_VERSION: str = "torchx_version"
LABEL_APP_NAME: str = "torchx_app_name"

DEFAULT_LOC: str = "us-central1"

# TODO Remove LOCATIONS list once Batch supports all locations
# or when there is an API to query locations supported by Batch
LOCATIONS: List[str] = [
    DEFAULT_LOC,
    "us-west1",
    "us-east1",
    "asia-southeast1",
    "europe-north1",
    "europe-west6",
]

BATCH_LOGGER_NAME = "batch_task_logs"


@dataclass
class GCPBatchJob:
    name: str
    project: str
    location: str
    job_def: "batch_v1.Job"

    def __str__(self) -> str:
        return yaml.dump(self.job_def)

    def __repr__(self) -> str:
        return str(self)


class GCPBatchOpts(TypedDict, total=False):
    project: Optional[str]
    location: Optional[str]


class GCPBatchScheduler(Scheduler[GCPBatchOpts, AppDef, AppDryRunInfo[GCPBatchJob]]):
    """
    GCPBatchScheduler is a TorchX scheduling interface to GCP Batch.

    .. code-block:: bash

        $ pip install torchx[gcp_batch]
        $ torchx run --scheduler gcp_batch utils.echo --msg hello
        # This launches a job with app handle like gcp_batch://torchx/project:location:app_id1234 and prints it
        $ torchx status gcp_batch://torchx/project:location:app_id1234
        ...

    Authentication is loaded from the environment using the gcloud credential handling.

    **Config Options**

    .. runopts::
        class: torchx.schedulers.gcp_batch_scheduler.create_scheduler

    **Compatibility**

    .. compatibility::
        type: scheduler
        features:
            cancel: true
            logs: true
            describe: true
            distributed: true
            workspaces: false
            mounts: false
            elasticity: false

    """

    def __init__(
        self,
        session_name: str,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        client: Optional[Any] = None,
    ) -> None:
        # NOTE: make sure any new init options are supported in create_scheduler(...)
        Scheduler.__init__(self, "gcp_batch", session_name)
        # pyre-fixme[4]: Attribute annotation cannot be `Any`.
        self.__client = client

    @property
    # pyre-fixme[3]: Return annotation cannot be `Any`.
    def _client(self) -> Any:
        from google.api_core import gapic_v1
        from google.cloud import batch_v1

        c = self.__client
        if c is None:
            client_info = gapic_v1.client_info.ClientInfo(
                user_agent=f"TorchX/{torchx.__version__}"
            )
            c = self.__client = batch_v1.BatchServiceClient(client_info=client_info)
        return c

    def schedule(self, dryrun_info: AppDryRunInfo[GCPBatchJob]) -> str:
        from google.cloud import batch_v1

        req = dryrun_info.request
        assert req is not None, f"{dryrun_info} missing request"

        request = batch_v1.CreateJobRequest(
            parent=f"projects/{req.project}/locations/{req.location}",
            job=req.job_def,
            job_id=req.name,
        )

        response = self._client.create_job(request=request)
        return f"{req.project}:{req.location}:{req.name}"

    def _app_to_job(self, app: AppDef) -> "batch_v1.Job":
        from google.cloud import batch_v1

        name = normalize_str(make_unique(app.name))

        taskGroups = []
        allocationPolicy = None

        # 1. Convert role to task
        # TODO implement retry_policy, mount conversion
        # NOTE: Supports only one role for now as GCP Batch supports only one TaskGroup
        # which is ok to start with as most components have only one role
        for role_idx, role in enumerate(app.roles):
            values = macros.Values(
                img_root="",
                app_id=name,
                replica_id=str(0),
                rank0_env=("BATCH_MAIN_NODE_HOSTNAME"),
            )
            role_dict = values.apply(role)
            role_dict.env["TORCHX_ROLE_IDX"] = str(role_idx)
            role_dict.env["TORCHX_ROLE_NAME"] = str(role.name)

            resource = role_dict.resource
            res = batch_v1.ComputeResource()
            cpu = resource.cpu
            if cpu <= 0:
                cpu = 1
            MILLI = 1000
            res.cpu_milli = cpu * MILLI
            memMB = resource.memMB
            if memMB < 0:
                raise ValueError(
                    f"memMB should to be set to a positive value, got {memMB}"
                )
            res.memory_mib = memMB

            # TODO support named resources
            # Using v100 as default GPU type as a100 does not allow changing count for now
            # TODO See if there is a better default GPU type
            if resource.gpu > 0:
                if resource.gpu not in GPU_COUNT_TO_TYPE:
                    raise ValueError(
                        f"gpu should to be set to one of these values: {GPU_COUNT_TO_TYPE.keys()}"
                    )
                machineType = GPU_COUNT_TO_TYPE[resource.gpu]
                allocationPolicy = batch_v1.AllocationPolicy(
                    instances=[
                        batch_v1.AllocationPolicy.InstancePolicyOrTemplate(
                            install_gpu_drivers=True,
                            policy=batch_v1.AllocationPolicy.InstancePolicy(
                                machine_type=machineType,
                            ),
                        )
                    ],
                )
                print(f"Using GPUs of type: {machineType}")

            # Configure host firewall rules to accept ingress communication
            config_network_runnable = batch_v1.Runnable(
                script=batch_v1.Runnable.Script(
                    text="/sbin/iptables -A INPUT -j ACCEPT"
                )
            )

            runnable = batch_v1.Runnable(
                container=batch_v1.Runnable.Container(
                    image_uri=role_dict.image,
                    commands=[role_dict.entrypoint] + role_dict.args,
                    entrypoint="",
                    # Configure docker to use the host network stack to communicate with containers/other hosts in the same network
                    options="--net host",
                )
            )

            ts = batch_v1.TaskSpec(
                runnables=[config_network_runnable, runnable],
                environment=batch_v1.Environment(variables=role_dict.env),
                max_retry_count=role_dict.max_retries,
                compute_resource=res,
            )

            task_env = [
                batch_v1.Environment(variables={"TORCHX_REPLICA_IDX": str(i)})
                for i in range(role_dict.num_replicas)
            ]

            tg = batch_v1.TaskGroup(
                task_spec=ts,
                task_count=role_dict.num_replicas,
                task_count_per_node=1,
                task_environments=task_env,
                require_hosts_file=True,
            )
            taskGroups.append(tg)

        # 2. Convert AppDef to Job
        job = batch_v1.Job(
            name=name,
            task_groups=taskGroups,
            allocation_policy=allocationPolicy,
            logs_policy=batch_v1.LogsPolicy(
                destination=batch_v1.LogsPolicy.Destination.CLOUD_LOGGING,
            ),
            # NOTE: GCP Batch does not allow label names with "."
            labels={
                LABEL_VERSION: torchx.__version__.replace(".", "-"),
                LABEL_APP_NAME: name,
            },
        )
        return job

    def _get_project(self) -> str:
        from google.cloud import runtimeconfig

        return runtimeconfig.Client().project

    def _submit_dryrun(
        self, app: AppDef, cfg: GCPBatchOpts
    ) -> AppDryRunInfo[GCPBatchJob]:
        proj = cfg.get("project")
        if proj is None:
            proj = self._get_project()
        assert proj is not None and isinstance(proj, str), "project must be a str"

        loc = cfg.get("location")
        assert loc is not None and isinstance(loc, str), "location must be a str"

        job = self._app_to_job(app)

        # Convert JobDef + BatchOpts to GCPBatchJob
        req = GCPBatchJob(
            name=str(job.name),
            project=proj,
            location=loc,
            job_def=job,
        )

        return AppDryRunInfo(req, repr)

    def run_opts(self) -> runopts:
        opts = runopts()
        opts.add(
            "project",
            type_=str,
            help="Name of the GCP project. Defaults to the configured GCP project in the environment",
        )
        opts.add(
            "location",
            type_=str,
            default=DEFAULT_LOC,
            help=f"Name of the location to schedule the job in. Defaults to {DEFAULT_LOC}",
        )
        return opts

    def _app_id_to_job_full_name(self, app_id: str) -> str:
        """
        app_id format: f"{project}:{location}:{name}"
        job_full_name format: f"projects/{project}/locations/{location}/jobs/{name}"
        where 'name' was created uniquely for the job from the app name
        """
        app_id_splits = app_id.split(":")
        if len(app_id_splits) != 3:
            raise ValueError(f"app_id not in expected format: {app_id}")
        return f"projects/{app_id_splits[0]}/locations/{app_id_splits[1]}/jobs/{app_id_splits[2]}"

    def _get_job(self, app_id: str) -> "batch_v1.Job":
        from google.cloud import batch_v1

        job_name = self._app_id_to_job_full_name(app_id)
        request = batch_v1.GetJobRequest(
            name=job_name,
        )
        return self._client.get_job(request=request)

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        job = self._get_job(app_id)
        if job is None:
            print(f"app not found: {app_id}")
            return None

        gpu = 0
        if len(job.allocation_policy.instances) != 0:
            gpu_type = job.allocation_policy.instances[0].policy.machine_type
            gpu = GPU_TYPE_TO_COUNT[gpu_type]

        roles = {}
        for tg in job.task_groups:
            env = tg.task_spec.environment.variables
            role = env["TORCHX_ROLE_NAME"]
            container = tg.task_spec.runnables[1].container
            roles[role] = Role(
                name=role,
                num_replicas=tg.task_count,
                image=container.image_uri,
                entrypoint=container.commands[0],
                args=list(container.commands[1:]),
                resource=Resource(
                    cpu=int(tg.task_spec.compute_resource.cpu_milli / 1000),
                    memMB=tg.task_spec.compute_resource.memory_mib,
                    gpu=gpu,
                ),
                env=dict(env),
                max_retries=tg.task_spec.max_retry_count,
            )

        # Map job -> DescribeAppResponse
        # TODO map role/replica status
        desc = DescribeAppResponse(
            app_id=app_id,
            state=JOB_STATE[job.status.state.name],
            roles=list(roles.values()),
        )
        return desc

    def log_iter(
        self,
        app_id: str,
        role_name: str = "",
        k: int = 0,
        regex: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        should_tail: bool = False,
        streams: Optional[Stream] = None,
    ) -> Iterable[str]:
        if streams not in (None, Stream.COMBINED):
            raise ValueError("GCPBatchScheduler only supports COMBINED log stream")

        job = self._get_job(app_id)
        if not job:
            raise ValueError(f"app not found: {app_id}")

        job_uid = job.uid
        filters = [
            f"labels.job_uid={job_uid}",
            f"labels.task_id:{job_uid}-group0-{k}",
        ]

        if since is not None:
            filters.append(f'timestamp>="{str(since.isoformat())}"')
        else:
            # gcloud logger.list by default only returns logs in the last 24 hours
            # Since many ML jobs can run longer add timestamp filter to get all logs
            filters.append(f'timestamp>="{str(datetime.fromtimestamp(0).isoformat())}"')

        if until is not None:
            filters.append(f'timestamp<="{str(until.isoformat())}"')
        if regex is not None:
            filters.append(f'textPayload =~ "{regex}"')
        filter = " AND ".join(filters)
        return self._batch_log_iter(filter)

    def _batch_log_iter(self, filter: str) -> Iterable[str]:
        from google.cloud import logging

        logger = logging.Client().logger(BATCH_LOGGER_NAME)
        for entry in logger.list_entries(filter_=filter):
            yield entry.payload + "\n"

    def _job_full_name_to_app_id(self, job_full_name: str) -> str:
        """
        job_full_name format: f"projects/{project}/locations/{location}/jobs/{name}"
        app_id format: f"{project}:{location}:{name}"
        where 'name' was created uniquely for the job from the app name
        """
        job_name_splits = job_full_name.split("/")
        if len(job_name_splits) != 6:
            raise ValueError(f"job full name not in expected format: {job_full_name}")
        return f"{job_name_splits[1]}:{job_name_splits[3]}:{job_name_splits[5]}"

    def list(self) -> List[ListAppResponse]:
        all_jobs = []
        proj = self._get_project()
        for loc in LOCATIONS:
            jobs = self._client.list_jobs(parent=f"projects/{proj}/locations/{loc}")
            all_jobs += jobs
        all_jobs.sort(key=lambda job: job.create_time.timestamp(), reverse=True)
        return [
            ListAppResponse(
                app_id=self._job_full_name_to_app_id(job.name),
                state=JOB_STATE[job.status.state.name],
            )
            for job in all_jobs
        ]

    def _validate(self, app: AppDef, scheduler: str, cfg: GCPBatchOpts) -> None:
        # Skip validation step
        pass

    def _cancel_existing(self, app_id: str) -> None:
        from google.cloud import batch_v1

        job_name = self._app_id_to_job_full_name(app_id)
        request = batch_v1.DeleteJobRequest(
            name=job_name,
            reason="Killed via TorchX",
        )
        self._client.delete_job(request=request)


def create_scheduler(
    session_name: str,
    # pyre-fixme[2]: Parameter annotation cannot be `Any`.
    client: Optional[Any] = None,
    **kwargs: object,
) -> GCPBatchScheduler:
    return GCPBatchScheduler(
        session_name=session_name,
        client=client,
    )
