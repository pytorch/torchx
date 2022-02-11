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

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, Mapping, Optional, Any

import torchx
import yaml
from torchx.schedulers.api import (
    AppDryRunInfo,
    DescribeAppResponse,
    Scheduler,
    Stream,
    filter_regex,
)
from torchx.schedulers.ids import make_unique
from torchx.specs.api import (
    AppDef,
    AppState,
    Role,
    SchedulerBackend,
    macros,
    runopts,
    CfgVal,
)

JOB_STATE: Dict[str, AppState] = {
    "SUBMITTED": AppState.PENDING,
    "PENDING": AppState.PENDING,
    "RUNNABLE": AppState.PENDING,
    "STARTING": AppState.PENDING,
    "RUNNING": AppState.RUNNING,
    "SUCCEEDED": AppState.SUCCEEDED,
    "FAILED": AppState.FAILED,
}


def role_to_node_properties(idx: int, role: Role) -> Dict[str, object]:
    resource = role.resource
    reqs = []
    cpu = resource.cpu
    if cpu <= 0:
        cpu = 1
    reqs.append({"type": "VCPU", "value": str(cpu)})

    mem = resource.memMB
    if mem <= 0:
        mem = 1000
    reqs.append({"type": "MEMORY", "value": str(mem)})

    if resource.gpu >= 0:
        reqs.append({"type": "GPU", "value": str(resource.gpu)})

    container = {
        "command": [role.entrypoint] + role.args,
        "image": role.image,
        "environment": [{"name": k, "value": v} for k, v in role.env.items()],
        "resourceRequirements": reqs,
        "logConfiguration": {
            "logDriver": "awslogs",
        },
    }

    return {
        "targetNodes": str(idx),
        "container": container,
    }


@dataclass
class BatchJob:
    name: str
    queue: str
    job_def: Dict[str, object]

    def __str__(self) -> str:
        return yaml.dump(self.job_def)

    def __repr__(self) -> str:
        return str(self)


class AWSBatchScheduler(Scheduler):
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
        class: torchx.schedulers.aws_batch_scheduler.AWSBatchScheduler

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
            workspaces: false
    """

    def __init__(
        self,
        session_name: str,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        client: Optional[Any] = None,
        # pyre-fixme[2]: Parameter annotation cannot be `Any`.
        log_client: Optional[Any] = None,
    ) -> None:
        super().__init__("aws_batch", session_name)

        # pyre-fixme[4]: Attribute annotation cannot be `Any`.
        self.__client = client
        # pyre-fixme[4]: Attribute annotation cannot be `Any`.
        self.__log_client = log_client

    @property
    # pyre-fixme[3]: Return annotation cannot be `Any`.
    def _client(self) -> Any:
        if self.__client is None:
            import boto3

            self.__client = boto3.client("batch")
        return self.__client

    @property
    # pyre-fixme[3]: Return annotation cannot be `Any`.
    def _log_client(self) -> Any:
        if self.__log_client is None:
            import boto3

            self.__log_client = boto3.client("logs")
        return self.__log_client

    def schedule(self, dryrun_info: AppDryRunInfo[BatchJob]) -> str:
        cfg = dryrun_info._cfg
        assert cfg is not None, f"{dryrun_info} missing cfg"
        req = dryrun_info.request
        self._client.register_job_definition(**req.job_def)

        self._client.submit_job(
            jobName=req.name,
            jobQueue=req.queue,
            jobDefinition=req.name,
            tags=req.job_def["tags"],
        )

        return f"{req.queue}:{req.name}"

    def _submit_dryrun(
        self, app: AppDef, cfg: Mapping[str, CfgVal]
    ) -> AppDryRunInfo[BatchJob]:
        queue = cfg.get("queue")
        if not isinstance(queue, str):
            raise TypeError(f"config value 'queue' must be a string, got {queue}")
        name = make_unique(app.name)

        nodes = []

        for role_idx, role in enumerate(app.roles):
            for replica_id in range(role.num_replicas):
                values = macros.Values(
                    img_root="",
                    app_id=name,
                    replica_id=str(replica_id),
                )
                replica_role = values.apply(role)
                replica_role.env["TORCHX_ROLE_IDX"] = str(role_idx)
                replica_role.env["TORCHX_ROLE_NAME"] = str(role.name)
                replica_role.env["TORCHX_REPLICA_IDX"] = str(replica_id)
                nodes.append(role_to_node_properties(len(nodes), replica_role))

        req = BatchJob(
            name=name,
            queue=queue,
            job_def={
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
        )
        info = AppDryRunInfo(req, repr)
        info._app = app
        info._cfg = cfg
        return info

    def _validate(self, app: AppDef, scheduler: SchedulerBackend) -> None:
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
                yield event["message"]


def create_scheduler(session_name: str, **kwargs: object) -> AWSBatchScheduler:
    return AWSBatchScheduler(
        session_name=session_name,
    )
