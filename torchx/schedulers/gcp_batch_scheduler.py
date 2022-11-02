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

LABEL_VERSION: str = "torchx_version"
LABEL_APP_NAME: str = "torchx_app_name"

DEFAULT_LOC: str = "us-central1"

DEFAULT_GPU_TYPE = "nvidia-tesla-v100"
DEFAULT_GPU_MACHINE_TYPE = "n1-standard-8"


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
        from google.cloud import batch_v1

        c = self.__client
        if c is None:
            c = self.__client = batch_v1.BatchServiceClient()
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
                allocationPolicy = batch_v1.AllocationPolicy(
                    instances=[
                        batch_v1.AllocationPolicy.InstancePolicyOrTemplate(
                            policy=batch_v1.AllocationPolicy.InstancePolicy(
                                machine_type=DEFAULT_GPU_MACHINE_TYPE,
                                accelerators=[
                                    batch_v1.AllocationPolicy.Accelerator(
                                        type_=DEFAULT_GPU_TYPE,
                                        count=resource.gpu,
                                    )
                                ],
                            )
                        )
                    ],
                )
                print(f"Using {resource.gpu} GPUs of type {DEFAULT_GPU_TYPE}")

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

    def _submit_dryrun(
        self, app: AppDef, cfg: GCPBatchOpts
    ) -> AppDryRunInfo[GCPBatchJob]:
        from google.cloud import runtimeconfig

        proj = cfg.get("project")
        if proj is None:
            proj = runtimeconfig.Client().project
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
        opts.add("project", type_=str, help="Name of the GCP project")
        opts.add(
            "location",
            type_=str,
            default=DEFAULT_LOC,
            help="Name of the location to schedule the job in",
        )
        return opts

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        from google.cloud import batch_v1

        # 1. get project, location, job name from app_id
        proj, loc, name = app_id.split(":")

        # 2. Get the Batch job
        request = batch_v1.GetJobRequest(
            name=f"projects/{proj}/locations/{loc}/jobs/{name}",
        )
        job = self._client.get_job(request=request)

        # 3. Map job -> DescribeAppResponse
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

    def list(self) -> List[ListAppResponse]:
        # Create ListJobsRequest with parent str
        # Use list_job api
        # map ListJobsPager response to ListAppResponse and return it
        raise NotImplementedError()

    def _validate(self, app: AppDef, scheduler: str) -> None:
        # Skip validation step
        pass

    def _cancel_existing(self, app_id: str) -> None:
        # 1.create DeleteJobRequest
        # get job name from app_id
        # use cancel reason - killed via torchX
        # 2. Submit request
        raise NotImplementedError()


def create_scheduler(session_name: str, **kwargs: object) -> GCPBatchScheduler:
    return GCPBatchScheduler(
        session_name=session_name,
    )
