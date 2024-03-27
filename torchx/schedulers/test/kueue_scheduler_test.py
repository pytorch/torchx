# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import base64
import importlib
import sys
import unittest
from datetime import datetime
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import torchx
from torchx import schedulers, specs

# @manual=//torchx/schedulers:kueue_scheduler
from torchx.schedulers import kueue_scheduler
from torchx.schedulers.api import AppDryRunInfo, DescribeAppResponse, ListAppResponse
from torchx.schedulers.docker_scheduler import has_docker
from torchx.schedulers.kueue_scheduler import (
    app_to_resource,
    create_scheduler,
    Kueue,
    KueueOpts,
    KueueScheduler,
)
from torchx.specs import AppState
from torchx.util.role_to_pod import LABEL_INSTANCE_TYPE, role_to_pod

SKIP_DOCKER: bool = not has_docker()

TEST_KUBE_CONFIG: Dict[str, Any] = {
    "current-context": "default",
    "contexts": [
        {
            "name": "default",
            "context": {
                "cluster": "default",
                "user": "torchx_fake_token",
                "namespace": "default",
            },
        }
    ],
    "clusters": [{"name": "default", "cluster": {"server": "torchx_test_host"}}],
    "users": [
        {
            "name": "torchx_fake_token",
            "user": {
                "token": base64.standard_b64encode(
                    "torchx-test-token".encode()
                ).decode(),
                "username": "me",
                "password": "password1234",
            },
        }
    ],
}


def _test_app(num_replicas: int = 1) -> specs.AppDef:
    trainer_role = specs.Role(
        name="trainer_foo",
        image="pytorch/torchx:latest",
        entrypoint="main",
        args=[
            "--output-path",
            specs.macros.img_root,
            "--app-id",
            specs.macros.app_id,
            "--rank0-env",
            specs.macros.rank0_env,
        ],
        env={"FOO": "bar"},
        resource=specs.Resource(
            cpu=2,
            memMB=3000,
            gpu=4,
        ),
        port_map={"foo": 1234},
        num_replicas=num_replicas,
        max_retries=3,
        mounts=[
            specs.BindMount(src_path="/src", dst_path="/dst", read_only=True),
        ],
    )
    return specs.AppDef("test", roles=[trainer_role])


class KueueSchedulerTest(unittest.TestCase):
    def test_create_scheduler(self) -> None:
        client = MagicMock()
        docker_client = MagicMock
        scheduler = create_scheduler("foo", client=client, docker_client=docker_client)
        self.assertIsInstance(scheduler, kueue_scheduler.KueueScheduler)
        self.assertEqual(scheduler._docker_client, docker_client)
        self.assertEqual(scheduler._client, client)

    def test_app_to_resource_resolved_macros(self) -> None:
        app = _test_app()
        unique_app_name = "app-name-42"
        with patch("torchx.schedulers.kueue_scheduler.make_unique") as make_unique_ctx:
            make_unique_ctx.return_value = unique_app_name
            resource = app_to_resource(
                app, service_account=None, local_queue="default-kueue"
            )
            actual_cmd = (
                # pyre-ignore [16]
                resource["spec"]["template"]
                .spec.containers[0]
                .command
            )
            expected_cmd = [
                "main",
                "--output-path",
                "",
                "--app-id",
                unique_app_name,
                "--rank0-env",
                "TORCHX_RANK0_HOST",
            ]
            self.assertEqual(expected_cmd, actual_cmd)

    def test_restart_policy_not_set(self) -> None:
        app = _test_app()
        resource = app_to_resource(
            app, service_account=None, local_queue="default-kueue"
        )
        self.assertEqual(
            "Never",
            # pyre-ignore [16]
            resource["spec"]["template"].spec.restart_policy,
        )
        for role in app.roles:
            role.max_retries = 0
        resource = app_to_resource(
            app, service_account=None, local_queue="default-kueue"
        )
        self.assertFalse("restartPolicy" in resource["spec"])
        self.assertFalse("backoffLimit" in resource["spec"])

    def test_role_to_pod(self) -> None:
        from kubernetes.client.models import (
            V1Container,
            V1ContainerPort,
            V1EmptyDirVolumeSource,
            V1EnvVar,
            V1HostPathVolumeSource,
            V1ObjectMeta,
            V1Pod,
            V1PodSpec,
            V1ResourceRequirements,
            V1SecurityContext,
            V1Volume,
            V1VolumeMount,
        )

        app = _test_app()
        pod = role_to_pod("name", app.roles[0], service_account="srvacc")

        limits = {
            "cpu": "2000m",
            "memory": "3000M",
            "nvidia.com/gpu": "4",
        }
        requests = {
            "cpu": "1900m",
            "memory": "1976M",
            "nvidia.com/gpu": "4",
        }
        resources = V1ResourceRequirements(
            limits=limits,
            requests=requests,
        )
        container = V1Container(
            command=[
                "main",
                "--output-path",
                specs.macros.img_root,
                "--app-id",
                specs.macros.app_id,
                "--rank0-env",
                specs.macros.rank0_env,
            ],
            image="pytorch/torchx:latest",
            name="name",
            env=[V1EnvVar(name="FOO", value="bar")],
            resources=resources,
            ports=[V1ContainerPort(name="foo", container_port=1234)],
            security_context=V1SecurityContext(),
            volume_mounts=[
                V1VolumeMount(
                    name="dshm",
                    mount_path="/dev/shm",
                ),
                V1VolumeMount(
                    name="mount-0",
                    mount_path="/dst",
                    read_only=True,
                ),
            ],
        )
        want = V1Pod(
            spec=V1PodSpec(
                containers=[container],
                restart_policy="Never",
                service_account_name="srvacc",
                volumes=[
                    V1Volume(
                        name="dshm",
                        empty_dir=V1EmptyDirVolumeSource(
                            medium="Memory",
                        ),
                    ),
                    V1Volume(
                        name="mount-0",
                        host_path=V1HostPathVolumeSource(
                            path="/src",
                        ),
                    ),
                ],
                node_selector={},
            ),
            metadata=V1ObjectMeta(
                annotations={
                    "sidecar.istio.io/inject": "false",
                },
                labels={},
            ),
        )

        self.assertEqual(
            pod,
            want,
        )

    def test_submit_dryrun(self) -> None:
        cfg = KueueOpts({"namespace": "testnamespace", "local_queue": "default-kueue"})
        scheduler = create_scheduler("test")
        app = _test_app()
        with patch("torchx.schedulers.kueue_scheduler.make_unique") as make_unique_ctx:
            make_unique_ctx.return_value = "app-name-42"
            info = scheduler.submit_dryrun(app, cfg)

        resource = str(info.request)
        self.assertEqual(
            resource,
            f"""apiVersion: batch/v1
kind: Job
metadata:
  name: app-name-42
spec:
  backoffLimit: 3
  name: trainerfoo-0
  replicas: 1
  template:
    metadata:
      annotations:
        sidecar.istio.io/inject: 'false'
      labels:
        app.kubernetes.io/instance: app-name-42
        app.kubernetes.io/managed-by: torchx.pytorch.org
        app.kubernetes.io/name: test
        kueue.x-k8s.io/queue-name: default-kueue
        torchx.pytorch.org/app-name: test
        torchx.pytorch.org/replica-id: '0'
        torchx.pytorch.org/role-index: '0'
        torchx.pytorch.org/role-name: trainer_foo
        torchx.pytorch.org/version: {torchx.__version__}
    spec:
      containers:
      - command:
        - main
        - --output-path
        - ''
        - --app-id
        - app-name-42
        - --rank0-env
        - TORCHX_RANK0_HOST
        env:
        - name: FOO
          value: bar
        - name: TORCHX_RANK0_HOST
          value: localhost
        image: pytorch/torchx:latest
        name: trainerfoo-0
        ports:
        - containerPort: 1234
          name: foo
        resources:
          limits:
            cpu: 2000m
            memory: 3000M
            nvidia.com/gpu: '4'
          requests:
            cpu: 1900m
            memory: 1976M
            nvidia.com/gpu: '4'
        securityContext: {{}}
        volumeMounts:
        - mountPath: /dev/shm
          name: dshm
        - mountPath: /dst
          name: mount-0
          readOnly: true
      nodeSelector: {{}}
      restartPolicy: Never
      volumes:
      - emptyDir:
          medium: Memory
        name: dshm
      - hostPath:
          path: /src
        name: mount-0
""",
        )

    def test_volume_mounts(self) -> None:
        scheduler = create_scheduler("test")
        from kubernetes.client.models import (
            V1EmptyDirVolumeSource,
            V1PersistentVolumeClaimVolumeSource,
            V1Volume,
            V1VolumeMount,
        )

        role = specs.Role(
            name="foo",
            image="",
            mounts=[
                specs.VolumeMount(src="name", dst_path="/dst", read_only=True),
            ],
        )
        pod = role_to_pod("foo", role, service_account="")
        self.assertEqual(
            pod.spec.volumes,
            [
                V1Volume(
                    name="dshm",
                    empty_dir=V1EmptyDirVolumeSource(
                        medium="Memory",
                    ),
                ),
                V1Volume(
                    name="mount-0",
                    persistent_volume_claim=V1PersistentVolumeClaimVolumeSource(
                        claim_name="name",
                    ),
                ),
            ],
        )
        self.assertEqual(
            pod.spec.containers[0].volume_mounts,
            [
                V1VolumeMount(
                    name="dshm",
                    mount_path="/dev/shm",
                ),
                V1VolumeMount(
                    name="mount-0",
                    mount_path="/dst",
                    read_only=True,
                ),
            ],
        )

    def test_device_mounts(self) -> None:
        scheduler = create_scheduler("test")
        from kubernetes.client.models import (
            V1HostPathVolumeSource,
            V1Volume,
            V1VolumeMount,
        )

        role = specs.Role(
            name="foo",
            image="",
            mounts=[
                specs.DeviceMount(src_path="foo", dst_path="bar", permissions="rwm"),
                specs.DeviceMount(src_path="foo2", dst_path="bar2", permissions="r"),
            ],
        )
        pod = role_to_pod("foo", role, service_account="")
        self.assertEqual(
            pod.spec.volumes[1:],
            [
                V1Volume(
                    name="mount-0",
                    host_path=V1HostPathVolumeSource(
                        path="foo",
                    ),
                ),
                V1Volume(
                    name="mount-1",
                    host_path=V1HostPathVolumeSource(
                        path="foo2",
                    ),
                ),
            ],
        )
        self.assertEqual(
            pod.spec.containers[0].volume_mounts[1:],
            [
                V1VolumeMount(
                    name="mount-0",
                    mount_path="bar",
                    read_only=False,
                ),
                V1VolumeMount(
                    name="mount-1",
                    mount_path="bar2",
                    read_only=True,
                ),
            ],
        )
        self.assertTrue(pod.spec.containers[0].security_context.privileged)

    def test_resource_devices(self) -> None:
        scheduler = create_scheduler("test")

        role = specs.Role(
            name="foo",
            image="",
            resource=specs.Resource(
                cpu=2,
                memMB=3000,
                gpu=4,
                devices={
                    "vpc.amazonaws.com/efa": 4,
                },
            ),
        )
        pod = role_to_pod("foo", role, service_account="")
        self.assertEqual(
            pod.spec.containers[0].resources.limits,
            {
                "cpu": "2000m",
                "memory": "3000M",
                "nvidia.com/gpu": "4",
                "vpc.amazonaws.com/efa": "4",
            },
        )
        self.assertFalse(pod.spec.containers[0].security_context.privileged)

    def test_instance_type(self) -> None:
        scheduler = create_scheduler("test")
        role = specs.Role(
            name="foo",
            image="",
            mounts=[],
            resource=specs.Resource(
                cpu=4,
                memMB=4000,
                gpu=8,
                capabilities={
                    LABEL_INSTANCE_TYPE: "some_instance",
                },
            ),
        )
        pod = role_to_pod("foo", role, service_account="")
        self.assertEqual(
            pod.spec.node_selector,
            {
                "node.kubernetes.io/instance-type": "some_instance",
            },
        )

    def test_rank0_env(self) -> None:
        from kubernetes.client.models import V1EnvVar

        scheduler = create_scheduler("test")
        app = _test_app(num_replicas=2)
        cfg = KueueOpts({"namespace": "testnamespace", "local_queue": "default-kueue"})
        with patch("torchx.schedulers.kueue_scheduler.make_unique") as make_unique_ctx:
            make_unique_ctx.return_value = "app-name-42"
            info = scheduler.submit_dryrun(app, cfg)

        task = info.request.resource["spec"]

        container0 = task["template"].spec.containers[0]
        self.assertIn("KUEUE_TRAINERFOO_0_HOSTS", container0.command)
        self.assertIn(V1EnvVar(name="FOO", value="bar"), container0.env)

    def test_submit_dryrun_patch(self) -> None:
        scheduler = create_scheduler("test")
        app = _test_app()
        app.roles[0].image = "sha256:testhash"
        cfg = KueueOpts(
            {"image_repo": "example.com/some/repo", "local_queue": "default-kueue"}
        )
        with patch("torchx.schedulers.kueue_scheduler.make_unique") as make_unique_ctx:
            make_unique_ctx.return_value = "app-name-42"
            info = scheduler.submit_dryrun(app, cfg)

        self.assertIn("example.com/some/repo:testhash", str(info.request.resource))
        self.assertEqual(
            info.request.images_to_push,
            {
                "sha256:testhash": (
                    "example.com/some/repo",
                    "testhash",
                ),
            },
        )

    def test_submit_dryrun_service_account(self) -> None:
        scheduler = create_scheduler("test")
        self.assertIn("service_account", scheduler.run_opts()._opts)
        app = _test_app()
        cfg = KueueOpts(
            {
                "service_account": "srvacc",
                "local_queue": "default-kueue",
            }
        )
        info = scheduler.submit_dryrun(app, cfg)
        self.assertIn("'service_account_name': 'srvacc'", str(info.request.resource))

        del cfg["service_account"]
        info = scheduler.submit_dryrun(app, cfg)
        self.assertIn("service_account_name': None", str(info.request.resource))

    @patch("kubernetes.client.BatchV1Api.create_namespaced_job")
    def test_submit(self, create_namespaced_job: MagicMock) -> None:
        from kubernetes.client.models import V1Job, V1ObjectMeta

        job = V1Job(
            api_version="v1",
            metadata=V1ObjectMeta(name="testid", namespace="testnamespace"),
        )

        create_namespaced_job.return_value = job

        scheduler = create_scheduler("test")
        app = _test_app()
        cfg = KueueOpts(
            {
                "namespace": "testnamespace",
                "local_queue": "default-kueue",
            }
        )

        info = scheduler.submit_dryrun(app, cfg)
        id = scheduler.schedule(info)
        self.assertEqual(id, "testnamespace:testid")
        call = create_namespaced_job.call_args
        args, kwargs = call
        self.assertEqual(kwargs["namespace"], "testnamespace")
        self.assertEqual(kwargs["body"], info.request.resource)

    @patch("kubernetes.client.CustomObjectsApi.create_namespaced_custom_object")
    def test_submit_no_kueue_label(
        self, create_namespaced_custom_object: MagicMock
    ) -> None:
        create_namespaced_custom_object.return_value = {
            "metadata": {"name": "testid"},
        }
        scheduler = create_scheduler("test")
        app = _test_app()
        cfg = KueueOpts(
            {
                "namespace": "testnamespace",
            }
        )
        with self.assertRaises(AssertionError):
            scheduler.submit_dryrun(app, cfg)

    @patch("kubernetes.client.BatchV1Api.create_namespaced_job")
    def test_submit_job_name_conflict(self, create_namespaced_job: MagicMock) -> None:
        from kubernetes.client.rest import ApiException

        api_exc = ApiException(status=409, reason="Conflict")
        api_exc.body = '{"details":{"name": "test_job"}}'
        create_namespaced_job.side_effect = api_exc

        scheduler = create_scheduler("test")
        app = _test_app()
        cfg = KueueOpts(
            {
                "namespace": "testnamespace",
                "local_queue": "default-kueue",
            }
        )
        info = scheduler.submit_dryrun(app, cfg)
        with self.assertRaises(ValueError):
            scheduler.schedule(info)

    @patch("kubernetes.client.BatchV1Api.read_namespaced_job_status")
    def test_describe(self, read_namespaced_job_status: MagicMock) -> None:
        from kubernetes.client.models import (
            V1Job,
            V1JobCondition,
            V1JobStatus,
            V1ObjectMeta,
        )

        job = V1Job(
            api_version="v1",
            metadata=V1ObjectMeta(name="testid", namespace="testnamespace"),
            status=V1JobStatus(
                completed_indexes=None,
                completion_time=None,
                conditions=[
                    V1JobCondition(
                        last_probe_time="datetime.datetime(2024, 2, 9, 10, 9, 44, tzinfo=tzutc())",
                        last_transition_time="datetime.datetime(2024, 2, 9, 10, 9, 44, tzinfo=tzutc())",
                        message="Job resumed",
                        reason="JobResumed",
                        status="False",
                        type="Suspended",
                    ),
                    V1JobCondition(
                        last_probe_time="datetime.datetime(2024, 2, 9, 10, 11, 58, tzinfo=tzutc())",
                        last_transition_time="datetime.datetime(2024, 2, 9, 10, 11, 58, tzinfo=tzutc())",
                        message=None,
                        reason=None,
                        status=True,
                        type="Complete",
                    ),
                ],
                failed=None,
                ready=0,
                start_time="datetime.datetime(2024, 2, 9, 10, 9, 44, tzinfo=tzutc())",
                succeeded=1,
                uncounted_terminated_pods={"failed": None, "succeeded": None},
            ),
        )

        read_namespaced_job_status.return_value = job
        app_id = "testnamespace:testid"
        scheduler = create_scheduler("test")
        info = scheduler.describe(app_id)
        call = read_namespaced_job_status.call_args
        args, kwargs = call

        assert "testid" in args
        assert "testnamespace" in args

        self.assertEqual(
            info,
            DescribeAppResponse(
                app_id=app_id,
                state=specs.AppState.SUCCEEDED,
                roles_statuses=[
                    specs.RoleStatus(
                        "",
                        [
                            specs.ReplicaStatus(
                                id=0,
                                role="",
                                state=specs.AppState.RESUMED,
                                hostname="",
                            ),
                            specs.ReplicaStatus(
                                id=0,
                                role="",
                                state=specs.AppState.SUCCEEDED,
                                hostname="",
                            ),
                        ],
                    ),
                ],
                roles=[
                    specs.Role(name="", image="", num_replicas=2),
                ],
            ),
        )

    @patch("kubernetes.client.BatchV1Api.read_namespaced_job_status")
    def test_describe_unknown(
        self, get_namespaced_custom_object_status: MagicMock
    ) -> None:
        get_namespaced_custom_object_status.return_value = {}
        app_id = "testnamespace:testid"
        scheduler = create_scheduler("test")
        info = scheduler.describe(app_id)

        self.assertEqual(
            info,
            DescribeAppResponse(
                app_id=app_id,
                state=specs.AppState.UNKNOWN,
            ),
        )

    def test_runopts(self) -> None:
        scheduler = kueue_scheduler.create_scheduler("foo")
        runopts = scheduler.run_opts()
        self.assertEqual(
            set(runopts._opts.keys()),
            {
                "namespace",
                "local_queue",
                "image_repo",
                "service_account",
                "priority_class",
                "annotations",
            },
        )

    @patch("kubernetes.client.BatchV1Api.delete_namespaced_job")
    def test_cancel_existing(self, delete_namespaced_job: MagicMock) -> None:
        from kubernetes import client

        scheduler = create_scheduler("test")
        scheduler._cancel_existing("testnamespace:testjob")
        call = delete_namespaced_job.call_args
        args, kwargs = call
        self.assertEqual(
            kwargs,
            {
                "namespace": "testnamespace",
                "name": "testjob",
                "body": client.V1DeleteOptions(propagation_policy="Foreground"),
            },
        )

    @patch("kubernetes.client.CustomObjectsApi.list_namespaced_custom_object")
    def test_list(self, list_namespaced_custom_object: MagicMock) -> None:
        with patch(
            "torchx.schedulers.kueue_scheduler.KueueScheduler._get_active_context"
        ) as test_context:
            test_context.return_value = TEST_KUBE_CONFIG["contexts"][0]
            scheduler = create_scheduler("test")

            scheduler.list()
            call = list_namespaced_custom_object.call_args
            args, kwargs = call

            self.assertEqual(
                kwargs,
                {
                    "group": "batch",
                    "version": "v1",
                    "namespace": "default",
                    "plural": "job",
                    "timeout_seconds": 30,
                },
            )

    @patch("kubernetes.client.CustomObjectsApi.list_namespaced_custom_object")
    def test_list_values(self, list_namespaced_custom_object: MagicMock) -> None:
        list_namespaced_custom_object.return_value = {
            "apiVersion": "batch/v1",
            "items": [
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "metadata": {
                        "creationTimestamp": "2021-10-11T20:49:35Z",
                        "name": "cifar-trainer-something",
                        "namespace": "default",
                        "resourceVersion": "100000000",
                        "uid": "ab6a11d3-aaaa-aaaa-aaaa-88220d5190ee",
                    },
                    "status": {
                        "runningDuration": "3262h8m50.910883962s",
                        "state": {
                            "lastTransitionTime": "2021-10-11T20:52:08Z",
                            "phase": "Completed",
                        },
                        "succeeded": 2,
                    },
                },
                {
                    "apiVersion": "batch/v1",
                    "kind": "Job",
                    "metadata": {
                        "creationTimestamp": "2021-10-11T20:49:35Z",
                        "name": "test-trainer",
                        "namespace": "default",
                        "resourceVersion": "100000000",
                        "uid": "ab6a11d3-bbbb-bbbb-bbbb-88220d5190ee",
                    },
                    "status": {
                        "runningDuration": "3262h8m50.910883962s",
                        "state": {
                            "lastTransitionTime": "2021-10-11T20:52:08Z",
                            "phase": "Terminated",
                        },
                    },
                },
            ],
        }
        with patch(
            "torchx.schedulers.kueue_scheduler.KueueScheduler._get_active_context"
        ) as test_context:
            test_context.return_value = TEST_KUBE_CONFIG["contexts"][0]

            scheduler = create_scheduler("test")

            apps = scheduler.list()
            call = list_namespaced_custom_object.call_args
            args, kwargs = call

            self.assertEqual(
                kwargs,
                {
                    "group": "batch",
                    "version": "v1",
                    "namespace": "default",
                    "plural": "job",
                    "timeout_seconds": 30,
                },
            )
            self.assertEqual(
                apps,
                [
                    ListAppResponse(
                        app_id="default:cifar-trainer-something",
                        state=AppState.SUCCEEDED,
                    ),
                    ListAppResponse(
                        app_id="default:test-trainer", state=AppState.FAILED
                    ),
                ],
            )

    @patch("kubernetes.client.CustomObjectsApi.list_namespaced_custom_object")
    def test_list_failure(self, list_namespaced_custom_object: MagicMock) -> None:
        from kubernetes.client.rest import ApiException

        api_exc = ApiException(
            status=404, reason="Invalid kube-config file. No configuration found."
        )
        list_namespaced_custom_object.side_effect = api_exc
        with patch(
            "torchx.schedulers.kueue_scheduler.KueueScheduler._get_active_context"
        ) as test_context:
            test_context.return_value = TEST_KUBE_CONFIG["contexts"][0]
            scheduler = create_scheduler("test")
            with self.assertRaises(ApiException):
                scheduler.list()

    @patch(
        "torchx.schedulers.kueue_scheduler.get_pod_name_from_job",
        return_value="testjob-roleblah-1-0",
    )
    @patch("kubernetes.client.CoreV1Api.read_namespaced_pod_log")
    def test_log_iter(
        self, read_namespaced_pod_log: MagicMock, read_namespaced_job: MagicMock
    ) -> None:
        scheduler = create_scheduler("test")
        read_namespaced_pod_log.return_value = "foo reg\nfoo\nbar reg\n"
        read_namespaced_job.return_value = "testjob-roleblah-1-0"

        lines = scheduler.log_iter(
            app_id="testnamespace:testjob",
            role_name="role_blah",
            k=1,
            regex="reg",
            since=datetime.now(),
        )
        self.assertEqual(
            list(lines),
            [
                "foo reg\n",
                "bar reg\n",
            ],
        )
        call = read_namespaced_pod_log.call_args
        args, kwargs = call
        self.assertGreaterEqual(kwargs["since_seconds"], 0)
        del kwargs["since_seconds"]
        self.assertEqual(
            kwargs,
            {
                "namespace": "testnamespace",
                "name": "testjob-roleblah-1-0",
                "timestamps": True,
            },
        )

    def test_push_patches(self) -> None:
        client = MagicMock()
        scheduler = KueueScheduler(
            "foo",
            client=MagicMock(),
            docker_client=client,
        )

        job = Kueue(
            images_to_push={
                "sha256:testimage": ("repo.com/img", "testimage"),
            },
            resource={},
        )

        out = scheduler.schedule(AppDryRunInfo(job, repr))
        self.assertTrue(out)

        self.assertEqual(client.images.get.call_count, 1)
        self.assertEqual(client.images.get().tag.call_count, 1)
        self.assertEqual(client.images.push.call_count, 1)

    def test_min_replicas(self) -> None:
        app = _test_app(num_replicas=3)
        app.roles[0].min_replicas = 2
        resource = app_to_resource(
            app, service_account=None, local_queue="default-kueue"
        )

        # Modify the minAvailable property of each job in the resource
        for job in resource["spec"]:  # pyre-ignore[16]
            if "backoffLimit" not in job:
                continue
            resource["spec"]["minAvailable"] = max(
                0, resource["spec"]["backoffLimit"] - 1
            )

        min_available = resource["spec"]["minAvailable"]

        self.assertEqual(min_available, 2)

    def test_submit_dryrun_priority_class(self) -> None:
        scheduler = create_scheduler("test")
        self.assertIn("priority_class", scheduler.run_opts()._opts)
        app = _test_app()
        cfg = KueueOpts(
            {
                "namespace": "testnamespace",
                "local_queue": "default-kueue",
                "priority_class": "sample-priority",
            }
        )

        info = scheduler.submit_dryrun(app, cfg)
        self.assertIn(
            "'kueue.x-k8s.io/priority-class': 'sample-priority'",
            str(info.request.resource),
        )

        del cfg["priority_class"]
        info = scheduler.submit_dryrun(app, cfg)
        self.assertNotIn(
            "'kueue.x-k8s.io/priority-class': 'sample-priority'",
            str(info.request.resource),
        )

    def test_submit_dryrun_with_annotations(self) -> None:
        scheduler = create_scheduler("test")
        self.assertIn("annotations", scheduler.run_opts()._opts)
        app = _test_app()
        cfg = KueueOpts(
            {
                "namespace": "testnamespace",
                "local_queue": "default-kueue",
                "annotations": {"test": "true"},
            }
        )

        info = scheduler.submit_dryrun(app, cfg)
        self.assertIn(
            "'test': 'true'",
            str(info.request.resource["metadata"]["annotations"]),
        )

        del cfg["annotations"]
        info = scheduler.submit_dryrun(app, cfg)
        self.assertNotIn(
            "annotations",
            str(info.request.resource["metadata"]),
        )


class KueueSchedulerNoImportTest(unittest.TestCase):
    """
    KueueSchedulerNoImportTest tests the kubernetes scheduler behavior when
    Kubernetes is not available.
    """

    def setUp(self) -> None:
        # make all kubernetes modules unable to be imported
        for mod in list(sys.modules.keys()) + ["kubernetes"]:
            if mod.startswith("kubernetes"):
                sys.modules[mod] = None  # pyre-ignore

        # reload to ensure kueue_scheduler doesn't depend on them at import
        # time
        importlib.reload(kueue_scheduler)
        importlib.reload(schedulers)

    def tearDown(self) -> None:
        # reset all kubernetes modules we patched
        for mod in list(sys.modules.keys()):
            if mod.startswith("kubernetes"):
                del sys.modules[mod]
        # reimport kueue_scheduler to get to a clean state
        importlib.reload(kueue_scheduler)

    def test_runopts(self) -> None:
        scheduler = kueue_scheduler.create_scheduler("foo")
        self.assertIsNotNone(scheduler.run_opts())

    def test_describe(self) -> None:
        scheduler = kueue_scheduler.create_scheduler("foo")
        with self.assertRaises(ModuleNotFoundError):
            scheduler.describe("foo:bar")

    def test_dryrun(self) -> None:
        scheduler = kueue_scheduler.create_scheduler("foo")
        app = _test_app()
        cfg = KueueOpts({"namespace": "testnamespace", "local_queue": "default-kueue"})

        with self.assertRaises(ModuleNotFoundError):
            scheduler.submit_dryrun(app, cfg)
