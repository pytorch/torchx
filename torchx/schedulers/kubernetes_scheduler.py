#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import warnings
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Iterable

import yaml
from kubernetes import client, config
from kubernetes.client.models import (
    V1Pod,
    V1PodSpec,
    V1Container,
    V1EnvVar,
    V1ResourceRequirements,
    V1ContainerPort,
)
from torchx.schedulers.api import AppDryRunInfo, DescribeAppResponse, Scheduler
from torchx.specs.api import (
    AppState,
    ReplicaState,
    AppDef,
    Role,
    RunConfig,
    SchedulerBackend,
    macros,
    RetryPolicy,
    RoleStatus,
    ReplicaStatus,
    runopts,
)

RETRY_POLICIES: Mapping[str, Iterable[Mapping[str, str]]] = {
    RetryPolicy.REPLICA: [],
    RetryPolicy.APPLICATION: [
        {"event": "PodEvicted", "action": "RestartJob"},
        {"event": "PodFailed", "action": "RestartJob"},
    ],
}

JOB_STATE: Dict[str, AppState] = {
    # Pending is the phase that job is pending in the queue, waiting for
    # scheduling decision
    "Pending": AppState.PENDING,
    # Aborting is the phase that job is aborted, waiting for releasing pods
    "Aborting": AppState.RUNNING,
    # Aborted is the phase that job is aborted by user or error handling
    "Aborted": AppState.CANCELLED,
    # Running is the phase that minimal available tasks of Job are running
    "Running": AppState.RUNNING,
    # Restarting is the phase that the Job is restarted, waiting for pod
    # releasing and recreating
    "Restarting": AppState.RUNNING,
    # Completed is the phase that all tasks of Job are completed successfully
    "Completed": AppState.SUCCEEDED,
    # Terminating is the phase that the Job is terminated, waiting for releasing
    # pods
    "Terminating": AppState.RUNNING,
    # Teriminated is the phase that the job is finished unexpected, e.g. events
    "Terminated": AppState.FAILED,
}

TASK_STATE: Dict[str, ReplicaState] = {
    # Pending means the task is pending in the apiserver.
    "Pending": ReplicaState.PENDING,
    # Allocated means the scheduler assigns a host to it.
    "Allocated": ReplicaState.PENDING,
    # Pipelined means the scheduler assigns a host to wait for releasing
    # resource.
    "Pipelined": ReplicaState.PENDING,
    # Binding means the scheduler send Bind request to apiserver.
    "Binding": ReplicaState.PENDING,
    # Bound means the task/Pod bounds to a host.
    "Bound": ReplicaState.PENDING,
    # Running means a task is running on the host.
    "Running": ReplicaState.RUNNING,
    # Releasing means a task/pod is deleted.
    "Releasing": ReplicaState.RUNNING,
    # Succeeded means that all containers in the pod have voluntarily
    # terminated with a container exit code of 0, and the system is not
    # going to restart any of these containers.
    "Succeeded": ReplicaState.SUCCEEDED,
    # Failed means that all containers in the pod have terminated, and at
    # least one container has terminated in a failure (exited with a
    # non-zero exit code or was stopped by the system).
    "Failed": ReplicaState.FAILED,
    # Unknown means the status of task/pod is unknown to the scheduler.
    "Unknown": ReplicaState.UNKNOWN,
}


def sanitize_for_serialization(obj: object) -> object:
    api = client.ApiClient()
    return api.sanitize_for_serialization(obj)


def role_to_pod(name: str, role: Role) -> V1Pod:
    assert role.base_image is None, "base_image is not supported by Kubernetes"

    requests = {}

    resource = role.resource
    if resource.cpu >= 0:
        requests["cpu"] = f"{int(resource.cpu*1000)}m"
    if resource.memMB >= 0:
        requests["memory"] = f"{int(resource.memMB)}M"
    if resource.gpu >= 0:
        requests["nvidia.com/gpu"] = str(resource.gpu)

    resources = V1ResourceRequirements(
        limits=requests,
        requests=requests,
    )

    container = V1Container(
        command=[role.entrypoint] + role.args,
        image=role.image,
        name=name,
        env=[
            V1EnvVar(
                name=name,
                value=value,
            )
            for name, value in role.env.items()
        ],
        resources=resources,
        ports=[
            V1ContainerPort(
                name=name,
                container_port=port,
            )
            for name, port in role.port_map.items()
        ],
    )
    return V1Pod(
        spec=V1PodSpec(
            containers=[container],
            restart_policy="Never",
        ),
    )


@dataclass
class KubernetesJob:
    resource: Dict[str, object]

    def __str__(self) -> str:
        return yaml.dump(sanitize_for_serialization(self.resource))

    def __repr__(self) -> str:
        return str(self)


class KubernetesScheduler(Scheduler):
    """
    KubernetesScheduler is a TorchX scheduling interface to Kubernetes.

    Important: Volcano is required to be installed on the Kubernetes cluster.
    TorchX requires gang scheduling for multi-replica/multi-role execution
    and Volcano is currently the only supported scheduler with Kubernetes.
    For installation instructions see: https://github.com/volcano-sh/volcano

    .. code-block:: bash

        $ torchx run --scheduler kubernetes --scheduler_opts namespace=default,queue=test utils.echo --msg hello
        kubernetes://torchx_user/1234
        $ torchx status kubernetes://torchx_user/1234
        ...
    """

    def __init__(
        self, session_name: str, client: Optional[client.ApiClient] = None
    ) -> None:
        super().__init__("kubernetes", session_name)

        self._client = client

    def _custom_objects_api(self) -> client.CustomObjectsApi:
        if self._client is None:
            configuration = client.Configuration()
            try:
                config.load_kube_config(client_configuration=configuration)
            except config.ConfigException as e:
                warnings.warn(f"failed to load kube config: {e}")

            self._client = client.ApiClient(configuration)

        return client.CustomObjectsApi(self._client)

    def schedule(self, dryrun_info: AppDryRunInfo[KubernetesJob]) -> str:
        cfg = dryrun_info._cfg
        assert cfg is not None, f"{dryrun_info} missing cfg"
        namespace = cfg.get("namespace") or "default"
        resource = dryrun_info.request.resource

        resp = self._custom_objects_api().create_namespaced_custom_object(
            group="batch.volcano.sh",
            version="v1alpha1",
            namespace=namespace,
            plural="jobs",
            body=resource,
        )
        return f'{namespace}:{resp["metadata"]["name"]}'

    def _submit_dryrun(
        self, app: AppDef, cfg: RunConfig
    ) -> AppDryRunInfo[KubernetesJob]:
        queue = cfg.get("queue")
        tasks = []
        for i, role in enumerate(app.roles):
            for replica_id in range(role.num_replicas):
                values = macros.Values(
                    img_root="",
                    app_id=macros.app_id,
                    replica_id=str(replica_id),
                )
                name = f"{role.name}-{replica_id}"
                replica_role = values.apply(role)
                pod = role_to_pod(name, replica_role)
                tasks.append(
                    {
                        "replicas": 1,
                        "name": name,
                        "template": pod,
                        "maxRetry": role.max_retries,
                        "policies": RETRY_POLICIES[role.retry_policy],
                    }
                )

        job_retries = min(role.max_retries for role in app.roles)
        resource: Dict[str, object] = {
            "apiVersion": "batch.volcano.sh/v1alpha1",
            "kind": "Job",
            "metadata": {"generateName": f"{app.name}-"},
            "spec": {
                "schedulerName": "volcano",
                "queue": queue,
                "tasks": tasks,
                "maxRetry": job_retries,
            },
        }
        req = KubernetesJob(resource=resource)
        info = AppDryRunInfo(req, repr)
        info._app = app
        info._cfg = cfg
        return info

    def _validate(self, app: AppDef, scheduler: SchedulerBackend) -> None:
        # Skip validation step
        pass

    def _cancel_existing(self, app_id: str) -> None:
        pass

    def run_opts(self) -> runopts:
        opts = runopts()
        opts.add(
            "namespace",
            type_=str,
            help="Kubernetes namespace to schedule job in",
            default="default",
        )
        opts.add(
            "queue", type_=str, help="Volcano queue to schedule job in", required=True
        )
        return opts

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        namespace, name = app_id.split(":")
        roles = {}
        resp = self._custom_objects_api().get_namespaced_custom_object_status(
            group="batch.volcano.sh",
            version="v1alpha1",
            namespace=namespace,
            plural="jobs",
            name=name,
        )
        status = resp["status"]

        state_str = status["state"]["phase"]
        app_state = JOB_STATE[state_str]

        TASK_STATUS_COUNT = "taskStatusCount"

        if TASK_STATUS_COUNT in status:
            for name, status in status[TASK_STATUS_COUNT].items():
                role, idx = name.split("-")

                state_str = next(iter(status["phase"].keys()))
                state = TASK_STATE[state_str]

                if role not in roles:
                    roles[role] = RoleStatus(role, [])
                roles[role].replicas.append(
                    ReplicaStatus(id=int(idx), role=role, state=state, hostname="")
                )
        return DescribeAppResponse(
            app_id=app_id,
            roles_statuses=list(roles.values()),
            state=app_state,
        )


def create_scheduler(session_name: str, **kwargs: Any) -> KubernetesScheduler:
    return KubernetesScheduler(
        session_name=session_name,
    )
