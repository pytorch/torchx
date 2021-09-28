#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, Mapping, Optional, Iterable

import yaml

if TYPE_CHECKING:
    from kubernetes.client import ApiClient, CustomObjectsApi
    from kubernetes.client.models import (  # noqa: F401 imported but unused
        V1Pod,
        V1PodSpec,
        V1Container,
        V1EnvVar,
        V1ResourceRequirements,
        V1ContainerPort,
    )
    from kubernetes.client.rest import ApiException

import torchx
from torchx.schedulers.api import (
    AppDryRunInfo,
    DescribeAppResponse,
    Scheduler,
    filter_regex,
)
from torchx.schedulers.ids import make_unique
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

logger: logging.Logger = logging.getLogger(__name__)


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
    "Failed": ReplicaState.FAILED,
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

LABEL_VERSION = "torchx.pytorch.org/version"
LABEL_APP_NAME = "torchx.pytorch.org/app-name"
LABEL_ROLE_INDEX = "torchx.pytorch.org/role-index"
LABEL_ROLE_NAME = "torchx.pytorch.org/role-name"
LABEL_REPLICA_ID = "torchx.pytorch.org/replica-id"

ANNOTATION_ISTIO_SIDECAR = "sidecar.istio.io/inject"


def sanitize_for_serialization(obj: object) -> object:
    from kubernetes import client

    api = client.ApiClient()
    return api.sanitize_for_serialization(obj)


def role_to_pod(name: str, role: Role) -> "V1Pod":
    from kubernetes.client.models import (  # noqa: F811 redefinition of unused
        V1Pod,
        V1PodSpec,
        V1Container,
        V1EnvVar,
        V1ResourceRequirements,
        V1ContainerPort,
        V1ObjectMeta,
    )

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
        metadata=V1ObjectMeta(
            annotations={
                # Disable the istio sidecar as it prevents the containers from
                # exiting once finished.
                ANNOTATION_ISTIO_SIDECAR: "false",
            },
            labels={},
        ),
    )


def app_to_resource(app: AppDef, queue: str) -> Dict[str, object]:
    """
    app_to_resource creates a volcano job kubernetes resource definition from
    the provided AppDef. The resource definition can be used to launch the
    app on Kubernetes.

    To support macros we generate one task per replica instead of using the
    volcano `replicas` field since macros change the arguments on a per
    replica basis.

    Volcano has two levels of retries: one at the task level and one at the
    job level. When using the APPLICATION retry policy, the job level retry
    count is set to the minimum of the max_retries of the roles.
    """
    tasks = []
    unique_app_id = make_unique(app.name)
    for role_idx, role in enumerate(app.roles):
        for replica_id in range(role.num_replicas):
            values = macros.Values(
                img_root="",
                app_id=unique_app_id,
                replica_id=str(replica_id),
            )
            name = f"{role.name}-{replica_id}"
            replica_role = values.apply(role)

            pod = role_to_pod(name, replica_role)
            pod.metadata.labels.update(pod_labels(app, role_idx, role, replica_id))

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
        "metadata": {"name": f"{unique_app_id}"},
        "spec": {
            "schedulerName": "volcano",
            "queue": queue,
            "tasks": tasks,
            "maxRetry": job_retries,
            "plugins": {
                "svc": [],
                "env": [],
            },
        },
    }
    return resource


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

    This has been confirmed to work with Volcano v1.3.0 and Kubernetes versions
    v1.18-1.21. See https://github.com/pytorch/torchx/issues/120 which is
    tracking Volcano support for Kubernetes v1.22.

    .. code-block:: bash

        $ pip install torchx[kubernetes]
        $ torchx run --scheduler kubernetes --scheduler_args namespace=default,queue=test utils.echo --image alpine:latest --msg hello
        kubernetes://torchx_user/1234
        $ torchx status kubernetes://torchx_user/1234
        ...

    .. compatibility::
        type: scheduler
        features:
            cancel: true
            logs: true
            distributed: true
            describe: |
                Partial support. KubernetesScheduler will return job and replica
                status but does not provide the complete original AppSpec.
    """

    def __init__(self, session_name: str, client: Optional["ApiClient"] = None) -> None:
        super().__init__("kubernetes", session_name)

        self._client = client

    def _api_client(self) -> "ApiClient":
        from kubernetes import client, config

        c = self._client
        if c is None:
            configuration = client.Configuration()
            try:
                config.load_kube_config(client_configuration=configuration)
            except config.ConfigException as e:
                warnings.warn(f"failed to load kube config: {e}")

            c = self._client = client.ApiClient(configuration)

        return c

    def _custom_objects_api(self) -> "CustomObjectsApi":
        from kubernetes import client

        return client.CustomObjectsApi(self._api_client())

    def _get_job_name_from_exception(self, e: "ApiException") -> Optional[str]:
        try:
            return json.loads(e.body)["details"]["name"]
        except Exception as e:
            logger.exception("Unable to retrieve job name, got exception", e)
            return None

    def schedule(self, dryrun_info: AppDryRunInfo[KubernetesJob]) -> str:
        from kubernetes.client.rest import ApiException

        cfg = dryrun_info._cfg
        assert cfg is not None, f"{dryrun_info} missing cfg"
        namespace = cfg.get("namespace") or "default"
        resource = dryrun_info.request.resource
        try:
            resp = self._custom_objects_api().create_namespaced_custom_object(
                group="batch.volcano.sh",
                version="v1alpha1",
                namespace=namespace,
                plural="jobs",
                body=resource,
            )
        except ApiException as e:
            if e.status == 409 and e.reason == "Conflict":
                job_name = self._get_job_name_from_exception(e)
                raise ValueError(
                    f"Job `{job_name}` already exists. This seems like a transient exception, try resubmitting job"
                ) from e
            else:
                raise

        return f'{namespace}:{resp["metadata"]["name"]}'

    def _submit_dryrun(
        self, app: AppDef, cfg: RunConfig
    ) -> AppDryRunInfo[KubernetesJob]:
        queue = cfg.get("queue")
        if not isinstance(queue, str):
            raise TypeError(f"config value 'queue' must be a string, got {queue}")
        resource = app_to_resource(app, queue)
        req = KubernetesJob(resource=resource)
        info = AppDryRunInfo(req, repr)
        info._app = app
        info._cfg = cfg
        return info

    def _validate(self, app: AppDef, scheduler: SchedulerBackend) -> None:
        # Skip validation step
        pass

    def _cancel_existing(self, app_id: str) -> None:
        namespace, name = app_id.split(":")
        self._custom_objects_api().delete_namespaced_custom_object(
            group="batch.volcano.sh",
            version="v1alpha1",
            namespace=namespace,
            plural="jobs",
            name=name,
        )

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
        roles_statuses = {}
        resp = self._custom_objects_api().get_namespaced_custom_object_status(
            group="batch.volcano.sh",
            version="v1alpha1",
            namespace=namespace,
            plural="jobs",
            name=name,
        )
        status = resp.get("status")
        if status:
            state_str = status["state"]["phase"]
            app_state = JOB_STATE[state_str]

            TASK_STATUS_COUNT = "taskStatusCount"

            if TASK_STATUS_COUNT in status:
                for name, status in status[TASK_STATUS_COUNT].items():
                    role, _, idx = name.rpartition("-")

                    state_str = next(iter(status["phase"].keys()))
                    state = TASK_STATE[state_str]

                    if role not in roles:
                        roles[role] = Role(name=role, num_replicas=0, image="")
                        roles_statuses[role] = RoleStatus(role, [])
                    roles[role].num_replicas += 1
                    roles_statuses[role].replicas.append(
                        ReplicaStatus(id=int(idx), role=role, state=state, hostname="")
                    )
        else:
            app_state = AppState.UNKNOWN
        return DescribeAppResponse(
            app_id=app_id,
            roles=list(roles.values()),
            roles_statuses=list(roles_statuses.values()),
            state=app_state,
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
    ) -> Iterable[str]:
        assert until is None, "kubernetes API doesn't support until"

        from kubernetes import client, watch

        namespace, name = app_id.split(":")

        pod_name = f"{name}-{role_name}-{k}-0"

        args: Dict[str, object] = {
            "name": pod_name,
            "namespace": namespace,
            "timestamps": True,
        }
        if since is not None:
            args["since_seconds"] = (datetime.now() - since).total_seconds()

        core_api = client.CoreV1Api(self._api_client())
        if should_tail:
            w = watch.Watch()
            iterator = w.stream(core_api.read_namespaced_pod_log, **args)
        else:
            resp = core_api.read_namespaced_pod_log(**args)
            iterator = resp.strip().split("\n")

        if regex:
            return filter_regex(regex, iterator)
        else:
            return iterator


def create_scheduler(session_name: str, **kwargs: Any) -> KubernetesScheduler:
    return KubernetesScheduler(
        session_name=session_name,
    )


def pod_labels(
    app: AppDef, role_idx: int, role: Role, replica_id: int
) -> Dict[str, str]:
    return {
        LABEL_VERSION: torchx.__version__,
        LABEL_APP_NAME: app.name,
        LABEL_ROLE_INDEX: str(role_idx),
        LABEL_ROLE_NAME: role.name,
        LABEL_REPLICA_ID: str(replica_id),
    }
