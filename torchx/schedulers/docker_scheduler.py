# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import fnmatch
import io
import logging
import os.path
import posixpath
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import (
    Mapping,
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    Optional,
    List,
    Union,
    IO,
)

import fsspec
import torchx
import yaml
from torchx.schedulers.api import (
    AppDryRunInfo,
    DescribeAppResponse,
    WorkspaceScheduler,
    filter_regex,
    Stream,
)
from torchx.schedulers.ids import make_unique
from torchx.specs.api import (
    AppDef,
    AppState,
    ReplicaStatus,
    Role,
    RoleStatus,
    SchedulerBackend,
    macros,
    runopts,
    is_terminal,
    CfgVal,
)

if TYPE_CHECKING:
    from docker import DockerClient
    from docker.models.containers import Container

log: logging.Logger = logging.getLogger(__name__)

CONTAINER_STATE: Dict[str, AppState] = {
    "created": AppState.SUBMITTED,
    "restarting": AppState.PENDING,
    "running": AppState.RUNNING,
    "paused": AppState.PENDING,
    "removing": AppState.PENDING,
    "dead": AppState.FAILED,
}


@dataclass
class DockerContainer:
    image: str
    command: List[str]
    kwargs: Dict[str, object]


@dataclass
class DockerJob:
    app_id: str
    containers: List[DockerContainer]

    def __str__(self) -> str:
        return yaml.dump(self.containers)

    def __repr__(self) -> str:
        return str(self)


LABEL_VERSION = "torchx.pytorch.org/version"
LABEL_APP_ID = "torchx.pytorch.org/app-id"
LABEL_ROLE_NAME = "torchx.pytorch.org/role-name"
LABEL_REPLICA_ID = "torchx.pytorch.org/replica-id"

NETWORK = "torchx"


def has_docker() -> bool:
    try:
        import docker

        docker.from_env()
        return True
    except (ImportError, docker.errors.DockerException):
        return False


class DockerScheduler(WorkspaceScheduler):
    """
    DockerScheduler is a TorchX scheduling interface to Docker.

    This is exposed via the scheduler `local_docker`.

    This scheduler runs the provided app via the local docker runtime using the
    specified images in the AppDef. Docker must be installed and running.  This
    provides the closest environment to schedulers that natively use Docker such
    as Kubernetes.

    .. note:: docker doesn't provide gang scheduling mechanisms. If one replica
              in a job fails, only that replica will be restarted.



    .. compatibility::
        type: scheduler
        features:
            cancel: true
            logs: true
            distributed: true
            describe: |
                Partial support. DockerScheduler will return job and replica
                status but does not provide the complete original AppSpec.
    """

    def __init__(self, session_name: str) -> None:
        super().__init__("docker", session_name)

        self.__client: Optional["DockerClient"] = None

    def _client(self) -> "DockerClient":
        if not self.__client:
            import docker

            self.__client = docker.from_env()

        return self.__client

    def _ensure_network(self) -> None:
        import filelock
        from docker.errors import APIError

        client = self._client()
        lock_path = os.path.join(tempfile.gettempdir(), "torchx_docker_network_lock")

        # Docker networks.create check_duplicate has a race condition so we need
        # to do client side locking to ensure only one network is created.
        with filelock.FileLock(lock_path, timeout=10):
            try:
                client.networks.create(
                    name=NETWORK, driver="bridge", check_duplicate=True
                )
            except APIError as e:
                if "already exists" not in str(e):
                    raise

    def schedule(self, dryrun_info: AppDryRunInfo[DockerJob]) -> str:
        client = self._client()

        req = dryrun_info.request

        images = set()
        for container in req.containers:
            images.add(container.image)
        for image in images:
            log.info(f"Pulling container image: {image} (this may take a while)")
            try:
                client.images.pull(image)
            except Exception as e:
                log.warning(f"failed to pull image {image}, falling back to local: {e}")

        self._ensure_network()

        for container in req.containers:
            client.containers.run(
                container.image,
                container.command,
                detach=True,
                **container.kwargs,
            )

        return req.app_id

    def _submit_dryrun(
        self, app: AppDef, cfg: Mapping[str, CfgVal]
    ) -> AppDryRunInfo[DockerJob]:
        from docker.types import DeviceRequest

        default_env = {}
        copy_env = cfg.get("copy_env")
        if copy_env:
            assert isinstance(
                copy_env, list
            ), f"copy_env must be a list, got {copy_env}"
            keys = set()
            for pattern in copy_env:
                keys |= set(fnmatch.filter(os.environ.keys(), pattern))
            for k in keys:
                default_env[k] = os.environ[k]

        app_id = make_unique(app.name)
        req = DockerJob(app_id=app_id, containers=[])
        for role in app.roles:
            for replica_id in range(role.num_replicas):
                values = macros.Values(
                    img_root="",
                    app_id=app_id,
                    replica_id=str(replica_id),
                )
                replica_role = values.apply(role)
                name = f"{app_id}-{role.name}-{replica_id}"

                env = default_env.copy()
                if replica_role.env:
                    env.update(replica_role.env)

                c = DockerContainer(
                    image=replica_role.image,
                    command=[replica_role.entrypoint] + replica_role.args,
                    kwargs={
                        "name": name,
                        "environment": env,
                        "labels": {
                            LABEL_VERSION: torchx.__version__,
                            LABEL_APP_ID: app_id,
                            LABEL_ROLE_NAME: role.name,
                            LABEL_REPLICA_ID: str(replica_id),
                        },
                        "hostname": name,
                        "network": NETWORK,
                    },
                )
                if replica_role.max_retries > 0:
                    c.kwargs["restart_policy"] = {
                        "Name": "on-failure",
                        "MaximumRetryCount": replica_role.max_retries,
                    }
                resource = replica_role.resource
                if resource.memMB >= 0:
                    c.kwargs["mem_limit"] = f"{int(resource.memMB)}m"
                if resource.cpu >= 0:
                    c.kwargs["nano_cpus"] = int(resource.cpu * 1e9)
                if resource.gpu > 0:
                    # `compute` means a CUDA or OpenCL capable device.
                    # For more info:
                    # * https://github.com/docker/docker-py/blob/master/docker/types/containers.py
                    # * https://github.com/NVIDIA/nvidia-container-runtime
                    c.kwargs["device_requests"] = [
                        DeviceRequest(
                            count=resource.gpu,
                            capabilities=[["compute"]],
                        )
                    ]
                req.containers.append(c)

        info = AppDryRunInfo(req, repr)
        info._app = app
        info._cfg = cfg
        return info

    def _validate(self, app: AppDef, scheduler: SchedulerBackend) -> None:
        # Skip validation step
        pass

    def _get_container(self, app_id: str, role: str, replica_id: int) -> "Container":
        client = self._client()
        containers = client.containers.list(
            all=True,
            filters={
                "label": [
                    f"{LABEL_APP_ID}={app_id}",
                    f"{LABEL_ROLE_NAME}={role}",
                    f"{LABEL_REPLICA_ID}={replica_id}",
                ]
            },
        )
        if len(containers) == 0:
            raise RuntimeError(
                f"failed to find container for {app_id}/{role}/{replica_id}"
            )
        elif len(containers) > 1:
            raise RuntimeError(
                f"found multiple containers for {app_id}/{role}/{replica_id}: {containers}"
            )
        return containers[0]

    def _get_containers(self, app_id: str) -> List["Container"]:
        client = self._client()
        return client.containers.list(
            all=True, filters={"label": f"{LABEL_APP_ID}={app_id}"}
        )

    def _cancel_existing(self, app_id: str) -> None:
        containers = self._get_containers(app_id)
        for container in containers:
            container.stop()

    def run_opts(self) -> runopts:
        opts = runopts()
        opts.add(
            "copy_env",
            type_=List[str],
            default=None,
            help="list of glob patterns of environment variables to copy if not set in AppDef. Ex: FOO_*",
        )
        return opts

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        roles = {}
        roles_statuses = {}

        states = []

        containers = self._get_containers(app_id)
        for container in containers:
            role = container.labels[LABEL_ROLE_NAME]
            replica_id = container.labels[LABEL_REPLICA_ID]

            if role not in roles:
                roles[role] = Role(
                    name=role,
                    num_replicas=0,
                    image=container.image,
                )
                roles_statuses[role] = RoleStatus(role, [])
            roles[role].num_replicas += 1

            if container.status == "exited":
                # docker doesn't have success/failed states -- we have to call
                # `wait()` to get the exit code to determine that
                status = container.wait(timeout=10)
                if status["StatusCode"] == 0:
                    state = AppState.SUCCEEDED
                else:
                    state = AppState.FAILED
            else:
                state = CONTAINER_STATE[container.status]

            roles_statuses[role].replicas.append(
                ReplicaStatus(
                    id=int(replica_id),
                    role=role,
                    state=state,
                    hostname=container.name,
                )
            )
            states.append(state)

        state = AppState.UNKNOWN
        if all(is_terminal(state) for state in states):
            if all(state == AppState.SUCCEEDED for state in states):
                state = AppState.SUCCEEDED
            else:
                state = AppState.FAILED
        else:
            state = next(state for state in states if not is_terminal(state))

        return DescribeAppResponse(
            app_id=app_id,
            roles=list(roles.values()),
            roles_statuses=list(roles_statuses.values()),
            state=state,
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
        c = self._get_container(app_id, role_name, k)

        logs = c.logs(
            since=since,
            until=until,
            stream=should_tail,
            stderr=streams != Stream.STDOUT,
            stdout=streams != Stream.STDERR,
        )

        if isinstance(logs, (bytes, str)):
            logs = _to_str(logs)

            if len(logs) == 0:
                logs = []
            else:
                logs = logs.split("\n")

        logs = map(_to_str, logs)

        if regex:
            return filter_regex(regex, logs)
        else:
            return logs

    def build_workspace_image(self, img: str, workspace: str) -> str:
        """
        build_workspace_image creates a new image with the files in workspace
        overlaid on top of it.

        Args:
            img: a Docker image to use as a base
            workspace: a fsspec path to a directory with contents to be overlaid

        Returns:
            The new Docker image ID.
        """
        return _build_container_from_workspace(self._client(), img, workspace)


def _to_str(a: Union[str, bytes]) -> str:
    if isinstance(a, bytes):
        a = a.decode("utf-8")
    if a.endswith("\n"):
        a = a[:-1]
    return a


def create_scheduler(session_name: str, **kwargs: Any) -> DockerScheduler:
    return DockerScheduler(
        session_name=session_name,
    )


def _copy_to_tarfile(workspace: str, tf: tarfile.TarFile) -> None:
    # TODO(d4l3k) implement docker ignore files

    fs, path = fsspec.core.url_to_fs(workspace)
    assert isinstance(path, str), "path must be str"

    for dir, dirs, files in fs.walk(path, detail=True):
        assert isinstance(dir, str), "path must be str"
        relpath = posixpath.relpath(dir, path)
        for file, info in files.items():
            with fs.open(info["name"], "rb") as f:
                tinfo = tarfile.TarInfo(posixpath.join(relpath, file))
                tinfo.size = info["size"]
                tf.addfile(tinfo, f)


def _build_context(img: str, workspace: str) -> IO[bytes]:
    # f is closed by parent, NamedTemporaryFile auto closes on GC
    f = tempfile.NamedTemporaryFile(  # noqa P201
        prefix="torchx-context",
        suffix=".tar",
    )
    dockerfile = bytes(f"FROM {img}\nCOPY . .\n", encoding="utf-8")
    with tarfile.open(fileobj=f, mode="w") as tf:
        info = tarfile.TarInfo("Dockerfile")
        info.size = len(dockerfile)
        tf.addfile(info, io.BytesIO(dockerfile))

        _copy_to_tarfile(workspace, tf)

    f.seek(0)
    return f


def _build_container_from_workspace(
    client: "DockerClient", img: str, workspace: str
) -> str:
    context = _build_context(img, workspace)

    try:
        image, logs = client.images.build(
            fileobj=context,
            custom_context=True,
            pull=True,
            rm=True,
            labels={
                LABEL_VERSION: torchx.__version__,
            },
        )
    finally:
        context.close()

    return image.id
