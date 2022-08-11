#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""

This contains the TorchX AWS Batch scheduler which can be used to run TorchX
components directly on AWS Batch.

This scheduler is in prototype stage and may change without notice.

Prerequisites
==============

You'll need to create an AWS Batch queue configured for multi-node parallel jobs.

See
https://docs.aws.amazon.com/batch/latest/userguide/Batch_GetStarted.html#first-run-step-2
for how to setup a job queue and compute environment. It needs to be backed by
EC2 for multi-node parallel jobs.

See
https://docs.aws.amazon.com/batch/latest/userguide/multi-node-parallel-jobs.html
for more information on distributed jobs.

If you want to use workspaces and container patching you'll also need to
configure a docker registry to store the patched containers with your changes
such as AWS ECR.

See
https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html#cli-create-repository
for how to create a image repository.
"""

import re
import threading
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Tuple,
    TYPE_CHECKING,
    TypeVar,
)

import torchx
import yaml
from torchx.schedulers.api import (
    AppDryRunInfo,
    DescribeAppResponse,
    filter_regex,
    ListAppResponse,
    Scheduler,
    Stream,
)
from torchx.schedulers.devices import get_device_mounts
from torchx.schedulers.ids import make_unique
from torchx.specs.api import (
    AppDef,
    AppState,
    BindMount,
    DeviceMount,
    macros,
    Role,
    runopts,
    VolumeMount,
)
from torchx.workspace.docker_workspace import DockerWorkspace
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from docker import DockerClient

JOB_STATE: Dict[str, AppState] = {
    "SUBMITTED": AppState.PENDING,
    "PENDING": AppState.PENDING,
    "RUNNABLE": AppState.PENDING,
    "STARTING": AppState.PENDING,
    "RUNNING": AppState.RUNNING,
    "SUCCEEDED": AppState.SUCCEEDED,
    "FAILED": AppState.FAILED,
}


def _role_to_node_properties(idx: int, role: Role) -> Dict[str, object]:
    resource = role.resource
    reqs = []
    cpu = resource.cpu
    if cpu <= 0:
        cpu = 1
    reqs.append({"type": "VCPU", "value": str(cpu)})

    memMB = resource.memMB
    if memMB < 0:
        raise ValueError(
            f"AWSBatchScheduler requires memMB to be set to a positive value, got {memMB}"
        )
    reqs.append({"type": "MEMORY", "value": str(memMB)})

    if resource.gpu > 0:
        reqs.append({"type": "GPU", "value": str(resource.gpu)})

    role.mounts += get_device_mounts(resource.devices)

    mount_points = []
    volumes = []
    devices = []
    for i, mount in enumerate(role.mounts):
        name = f"mount_{i}"
        if isinstance(mount, BindMount):
            volumes.append(
                {
                    "name": name,
                    "host": {
                        "sourcePath": mount.src_path,
                    },
                }
            )
            mount_points.append(
                {
                    "containerPath": mount.dst_path,
                    "readOnly": mount.read_only,
                    "sourceVolume": name,
                }
            )
        elif isinstance(mount, VolumeMount):
            volumes.append(
                {
                    "name": name,
                    "efsVolumeConfiguration": {
                        "fileSystemId": mount.src,
                    },
                }
            )
            mount_points.append(
                {
                    "containerPath": mount.dst_path,
                    "readOnly": mount.read_only,
                    "sourceVolume": name,
                }
            )
        elif isinstance(mount, DeviceMount):
            perm_map = {
                "r": "READ",
                "w": "WRITE",
                "m": "MKNOD",
            }
            devices.append(
                {
                    "hostPath": mount.src_path,
                    "containerPath": mount.dst_path,
                    "permissions": [perm_map[p] for p in mount.permissions],
                },
            )
        else:
            raise TypeError(f"unknown mount type {mount}")

    container = {
        "command": [role.entrypoint] + role.args,
        "image": role.image,
        "environment": [{"name": k, "value": v} for k, v in role.env.items()],
        "resourceRequirements": reqs,
        "linuxParameters": {
            # To support PyTorch dataloaders we need to set /dev/shm to larger
            # than the 64M default.
            "sharedMemorySize": memMB,
            "devices": devices,
        },
        "logConfiguration": {
            "logDriver": "awslogs",
        },
        "mountPoints": mount_points,
        "volumes": volumes,
    }

    return {
        "targetNodes": str(idx),
        "container": container,
    }


def _job_ui_url(job_arn: str) -> Optional[str]:
    match = re.match(
        "arn:aws:batch:([a-z-0-9]+):[0-9]+:job/([a-z-0-9]+)",
        job_arn,
    )
    if match is None:
        return None
    region = match.group(1)
    job_id = match.group(2)
    return f"https://{region}.console.aws.amazon.com/batch/home?region={region}#jobs/mnp-job/{job_id}"


@dataclass
class BatchJob:
    name: str
    queue: str
    share_id: Optional[str]
    job_def: Dict[str, object]
    images_to_push: Dict[str, Tuple[str, str]]

    def __str__(self) -> str:
        return yaml.dump(self.job_def)

    def __repr__(self) -> str:
        return str(self)


T = TypeVar("T")


def _thread_local_cache(f: Callable[[], T]) -> Callable[[], T]:
    local: threading.local = threading.local()
    key: str = "value"

    def wrapper() -> T:
        if key in local.__dict__:
            return local.__dict__[key]

        v = f()
        local.__dict__[key] = v
        return v

    return wrapper


@_thread_local_cache
def _local_session() -> "boto3.session.Session":
    import boto3.session

    return boto3.session.Session()


class AWSBatchOpts(TypedDict, total=False):
    queue: str
    image_repo: Optional[str]
    share_id: Optional[str]
    priority: Optional[int]


class AWSBatchScheduler(Scheduler[AWSBatchOpts], DockerWorkspace):
    """
    AWSBatchScheduler is a TorchX scheduling interface to AWS Batch.

    .. code-block:: bash

        $ pip install torchx[kubernetes]
        $ torchx run --scheduler aws_batch --scheduler_args queue=torchx utils.echo --image alpine:latest --msg hello
        aws_batch://torchx_user/1234
        $ torchx status aws_batch://torchx_user/1234
        ...

    Authentication is loaded from the environment using the ``boto3`` credential
    handling.

    **Config Options**

    .. runopts::
        class: torchx.schedulers.aws_batch_scheduler.create_scheduler

    **Mounts**

    This class supports bind mounting host directories, efs volumes and host
    devices.

    * bind mount: ``type=bind,src=<host path>,dst=<container path>[,readonly]``
    * efs volume: ``type=volume,src=<efs id>,dst=<container path>[,readonly]``
    * devices: ``type=device,src=/dev/infiniband/uverbs0,[dst=<container path>][,perm=rwm]``

    See :py:func:`torchx.specs.parse_mounts` for more info.

    For other filesystems such as FSx you can mount them onto the host and bind
    mount them into your job: https://aws.amazon.com/premiumsupport/knowledge-center/batch-fsx-lustre-file-system-mount/

    For Elastic Fabric Adapter (EFA) you'll need to use a device mount to mount
    them into the container: https://docs.aws.amazon.com/batch/latest/userguide/efa.html

    **Compatibility**

    .. compatibility::
        type: scheduler
        features:
            cancel: true
            logs: true
            distributed: true
            describe: |
                Partial support. AWSBatchScheduler will return job and replica
                status but does not provide the complete original AppSpec.
            workspaces: true
            mounts: true
    """

    def __init__(
        self,
        session_name: str,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        client: Optional[Any] = None,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        log_client: Optional[Any] = None,
        docker_client: Optional["DockerClient"] = None,
    ) -> None:
        Scheduler.__init__(self, "aws_batch", session_name)
        DockerWorkspace.__init__(self, docker_client)

        # pyre-fixme[4]: Attribute annotation cannot be `Any`.
        self.__client = client
        # pyre-fixme[4]: Attribute annotation cannot be `Any`.
        self.__log_client = log_client

    @property
    # pyre-fixme[3]: Return annotation cannot be `Any`.
    def _client(self) -> Any:
        if self.__client:
            return self.__client
        return _local_session().client("batch")

    @property
    # pyre-fixme[3]: Return annotation cannot be `Any`.
    def _log_client(self) -> Any:
        if self.__log_client:
            return self.__log_client
        return _local_session().client("logs")

    def schedule(self, dryrun_info: AppDryRunInfo[BatchJob]) -> str:
        cfg = dryrun_info._cfg
        assert cfg is not None, f"{dryrun_info} missing cfg"

        images_to_push = dryrun_info.request.images_to_push
        self._push_images(images_to_push)

        req = dryrun_info.request
        self._client.register_job_definition(**req.job_def)

        batch_job_req = {
            **{
                "jobName": req.name,
                "jobQueue": req.queue,
                "jobDefinition": req.name,
                "tags": req.job_def["tags"],
            },
            **({"shareIdentifier": req.share_id} if req.share_id is not None else {}),
        }
        self._client.submit_job(**batch_job_req)

        return f"{req.queue}:{req.name}"

    def _submit_dryrun(self, app: AppDef, cfg: AWSBatchOpts) -> AppDryRunInfo[BatchJob]:
        queue = cfg.get("queue")
        if not isinstance(queue, str):
            raise TypeError(f"config value 'queue' must be a string, got {queue}")

        share_id = cfg.get("share_id")
        priority = cfg.get("priority")
        if share_id is None and priority is not None:
            raise ValueError(
                "config value 'priority' takes no effect for job queues without a scheduling policy "
                "(implied by 'share_id' not being set)"
            )

        name_suffix = f"-{share_id}" if share_id is not None else ""
        name = make_unique(f"{app.name}{name_suffix}")

        # map any local images to the remote image
        images_to_push = self._update_app_images(app, cfg.get("image_repo"))

        nodes = []

        for role_idx, role in enumerate(app.roles):
            for replica_id in range(role.num_replicas):
                rank = len(nodes)
                values = macros.Values(
                    img_root="",
                    app_id=name,
                    replica_id=str(replica_id),
                    rank0_env=(
                        "TORCHX_RANK0_HOST"
                        if rank == 0
                        else "AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS"
                    ),
                )
                replica_role = values.apply(role)
                replica_role.env["TORCHX_ROLE_IDX"] = str(role_idx)
                replica_role.env["TORCHX_ROLE_NAME"] = str(role.name)
                replica_role.env["TORCHX_REPLICA_IDX"] = str(replica_id)
                if rank == 0:
                    # AWS_BATCH_JOB_MAIN_NODE_PRIVATE_IPV4_ADDRESS is only
                    # available on the child workers so we set the address to
                    # localhost for rank0.
                    # See: https://docs.aws.amazon.com/batch/latest/userguide/job_env_vars.html
                    replica_role.env["TORCHX_RANK0_HOST"] = "localhost"
                nodes.append(_role_to_node_properties(rank, replica_role))

        job_def = {
            **{
                "jobDefinitionName": name,
                "type": "multinode",
                "nodeProperties": {
                    "numNodes": len(nodes),
                    "mainNode": 0,
                    "nodeRangeProperties": nodes,
                },
                "retryStrategy": {
                    "attempts": max(max(role.max_retries for role in app.roles), 1),
                    "evaluateOnExit": [
                        {"onExitCode": "0", "action": "EXIT"},
                    ],
                },
                "tags": {
                    "torchx.pytorch.org/version": torchx.__version__,
                    "torchx.pytorch.org/app-name": app.name,
                },
            },
            **(
                {"schedulingPriority": priority if priority is not None else 0}
                if share_id is not None
                else {}
            ),
        }

        req = BatchJob(
            name=name,
            queue=queue,
            share_id=share_id,
            job_def=job_def,
            images_to_push=images_to_push,
        )
        info = AppDryRunInfo(req, repr)
        info._app = app
        # pyre-fixme: AppDryRunInfo
        info._cfg = cfg
        return info

    def _validate(self, app: AppDef, scheduler: str) -> None:
        # Skip validation step
        pass

    def _cancel_existing(self, app_id: str) -> None:
        job_id = self._get_job_id(app_id)
        self._client.terminate_job(
            jobId=job_id,
            reason="killed via torchx CLI",
        )

    def run_opts(self) -> runopts:
        opts = runopts()
        opts.add("queue", type_=str, help="queue to schedule job in", required=True)
        opts.add(
            "image_repo",
            type_=str,
            help="The image repository to use when pushing patched images, must have push access. Ex: example.com/your/container",
        )
        opts.add(
            "share_id",
            type_=str,
            help="The share identifier for the job. "
            "This must be set if and only if the job queue has a scheduling policy.",
        )
        opts.add(
            "priority",
            type_=int,
            help="The scheduling priority for the job within the context of share_id. "
            "Higher number (between 0 and 9999) means higher priority. "
            "This will only take effect if the job queue has a scheduling policy.",
        )
        return opts

    def _get_job_id(self, app_id: str) -> Optional[str]:
        queue, name = app_id.split(":")

        job_summary_list = self._client.list_jobs(
            jobQueue=queue,
            filters=[{"name": "JOB_NAME", "values": [name]}],
        )["jobSummaryList"]
        if len(job_summary_list) == 0:
            return None
        return job_summary_list[0]["jobArn"]

    def _get_job(
        self, app_id: str, rank: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        job_id = self._get_job_id(app_id)
        if not job_id:
            return None
        if rank is not None:
            job_id += f"#{rank}"
        jobs = self._client.describe_jobs(jobs=[job_id])["jobs"]
        if len(jobs) == 0:
            return None
        return jobs[0]

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        job = self._get_job(app_id)
        if job is None:
            return None

        # TODO: role statuses

        roles = {}
        nodes = job["nodeProperties"]["nodeRangeProperties"]
        for node in nodes:
            container = node["container"]
            env = {opt["name"]: opt["value"] for opt in container["environment"]}
            role = env["TORCHX_ROLE_NAME"]
            replica_id = int(env["TORCHX_REPLICA_IDX"])

            if role not in roles:
                roles[role] = Role(
                    name=role,
                    num_replicas=0,
                    image=container["image"],
                    entrypoint=container["command"][0],
                    args=container["command"][1:],
                    env=env,
                )
            roles[role].num_replicas += 1

        return DescribeAppResponse(
            app_id=app_id,
            state=JOB_STATE[job["status"]],
            roles=list(roles.values()),
            ui_url=_job_ui_url(job["jobArn"]),
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
        if streams not in (None, Stream.COMBINED):
            raise ValueError("AWSBatchScheduler only supports COMBINED log stream")

        job = self._get_job(app_id)
        if job is None:
            return []
        nodes = job["nodeProperties"]["nodeRangeProperties"]
        i = 0
        for i, node in enumerate(nodes):
            container = node["container"]
            env = {opt["name"]: opt["value"] for opt in container["environment"]}
            node_role = env["TORCHX_ROLE_NAME"]
            replica_id = int(env["TORCHX_REPLICA_IDX"])
            if role_name == node_role and k == replica_id:
                break

        job = self._get_job(app_id, rank=i)
        if not job:
            return []

        if "status" in job and job["status"] == "RUNNING":
            stream_name = job["container"]["logStreamName"]
        else:
            attempts = job["attempts"]
            if len(attempts) == 0:
                return []

            attempt = attempts[-1]
            container = attempt["container"]
            stream_name = container["logStreamName"]

        iterator = self._stream_events(stream_name, since=since, until=until)
        if regex:
            return filter_regex(regex, iterator)
        else:
            return iterator

    def list(self) -> List[ListAppResponse]:
        # TODO: get queue name input instead of iterating over all queues?
        resp = self._client.describe_job_queues()
        queue_names = [queue["jobQueueName"] for queue in resp["jobQueues"]]
        all_apps = []
        for qn in queue_names:
            apps_in_queue = self._list_by_queue(qn)
            all_apps += [
                ListAppResponse(
                    app_id=f"{qn}:{app['jobName']}", state=JOB_STATE[app["status"]]
                )
                for app in apps_in_queue
            ]
        return all_apps

    def _list_by_queue(self, queue_name: str) -> List[Dict[str, Any]]:
        # By default only running jobs are listed by batch/boto client's list_jobs API
        # When 'filters' parameter is specified, jobs with all statuses are listed
        # So use AFTER_CREATED_AT filter to list jobs in all statuses
        # milli_seconds_after_epoch can later be used to list jobs by timeframe
        milli_seconds_after_epoch = "1"
        return self._client.list_jobs(
            jobQueue=queue_name,
            filters=[
                {
                    "name": "AFTER_CREATED_AT",
                    "values": [
                        milli_seconds_after_epoch,
                    ],
                },
            ],
        )["jobSummaryList"]

    def _stream_events(
        self,
        stream_name: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> Iterable[str]:

        next_token = None

        while True:
            args = {}
            if next_token is not None:
                args["nextToken"] = next_token
            if until is not None:
                args["endTime"] = until.timestamp()
            if since is not None:
                args["startTime"] = since.timestamp()
            try:
                response = self._log_client.get_log_events(
                    logGroupName="/aws/batch/job",
                    logStreamName=stream_name,
                    limit=10000,
                    startFromHead=True,
                    **args,
                )
            except self._log_client.exceptions.ResourceNotFoundException:
                return []  # noqa: B901
            if response["nextForwardToken"] == next_token:
                break
            next_token = response["nextForwardToken"]

            for event in response["events"]:
                yield event["message"] + "\n"


def create_scheduler(session_name: str, **kwargs: object) -> AWSBatchScheduler:
    return AWSBatchScheduler(
        session_name=session_name,
    )
