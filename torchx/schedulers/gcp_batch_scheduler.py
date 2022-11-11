#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""

This contains the TorchX GCP Batch scheduler which can be used to run TorchX
components directly on GCP Batch.

This scheduler is in prototype stage and may change without notice.

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
from torchx.specs.api import AppDef, AppState, macros, runopts
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


class GCPBatchScheduler(Scheduler[GCPBatchOpts]):
    """
    GCPBatchScheduler is a TorchX scheduling interface to GCP Batch.

    .. code-block:: bash

        $ pip install torchx
        $ torchx run --scheduler gcp_batch utils.echo --msg hello
        gcp_batch://torchx_user/1234
        $ torchx status gcp_batch://torchx_user/1234
        ...

    Authentication is loaded from the environment using the gcloud credential handling.

    **Config Options**

    .. runopts::
        class: torchx.schedulers.gcp_batch_scheduler.create_scheduler

    **Compatibility**

    .. compatibility::
        type: scheduler
        features:
            describe: |
                Partial support. GCPBatchScheduler will return job status
                but does not provide the complete original AppSpec.

    """

    def __init__(
        self,
        session_name: str,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        client: Optional[Any] = None,
    ) -> None:
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
                # TODO set value for rank0_env: TORCHX_RANK0_HOST is a place holder for now
                rank0_env=("TORCHX_RANK0_HOST"),
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
            # pyre-ignore [8] : pyre gets confused even when types on both sides of = are int
            res.cpu_milli = cpu * MILLI
            memMB = resource.memMB
            if memMB < 0:
                raise ValueError(
                    f"memMB should to be set to a positive value, got {memMB}"
                )
            # pyre-ignore [8] : pyre gets confused even when types on both sides of = are int
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

            runnable = batch_v1.Runnable(
                container=batch_v1.Runnable.Container(
                    image_uri=role_dict.image,
                    commands=[role_dict.entrypoint] + role_dict.args,
                    entrypoint="",
                )
            )

            ts = batch_v1.TaskSpec(
                runnables=[runnable],
                environments=role_dict.env,
                max_retry_count=role_dict.max_retries,
                compute_resource=res,
            )

            tg = batch_v1.TaskGroup(
                task_spec=ts,
                task_count=role_dict.num_replicas,
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

        info = AppDryRunInfo(req, repr)
        info._app = app
        # pyre-fixme: AppDryRunInfo
        info._cfg = cfg
        return info

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

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        from google.cloud import batch_v1

        job_name = self._app_id_to_job_full_name(app_id)

        # 1. Get the Batch job
        request = batch_v1.GetJobRequest(
            name=job_name,
        )
        job = self._client.get_job(request=request)

        # 2. Map job -> DescribeAppResponse
        # TODO map job taskGroup to Role, map env vars etc
        return DescribeAppResponse(
            app_id=app_id,
            state=JOB_STATE[job.status.state.name],
        )

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

    def _validate(self, app: AppDef, scheduler: str) -> None:
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


def create_scheduler(session_name: str, **kwargs: object) -> GCPBatchScheduler:
    return GCPBatchScheduler(
        session_name=session_name,
    )
