# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib
import sys
import unittest
from datetime import datetime
from unittest.mock import MagicMock, patch

import torchx
from torchx import schedulers, specs

# @manual=//torchx/schedulers:kubernetes_scheduler
from torchx.schedulers import kubernetes_scheduler
from torchx.schedulers.api import DescribeAppResponse
from torchx.schedulers.kubernetes_scheduler import (
    app_to_resource,
    cleanup_str,
    create_scheduler,
    role_to_pod,
)


def _test_app() -> specs.AppDef:
    trainer_role = specs.Role(
        name="trainer",
        image="pytorch/torchx:latest",
        entrypoint="main",
        args=["--output-path", specs.macros.img_root, "--app-id", specs.macros.app_id],
        env={"FOO": "bar"},
        resource=specs.Resource(
            cpu=2,
            memMB=3000,
            gpu=4,
        ),
        port_map={"foo": 1234},
        num_replicas=1,
        max_retries=3,
    )

    return specs.AppDef("test", roles=[trainer_role])


class KubernetesSchedulerTest(unittest.TestCase):
    def test_create_scheduler(self) -> None:
        scheduler = create_scheduler("foo")
        self.assertIsInstance(scheduler, kubernetes_scheduler.KubernetesScheduler)

    def test_app_to_resource_resolved_macros(self) -> None:
        app = _test_app()
        unique_app_name = "app-name-42"
        with patch(
            "torchx.schedulers.kubernetes_scheduler.make_unique"
        ) as make_unique_ctx:
            make_unique_ctx.return_value = unique_app_name
            resource = app_to_resource(app, "test_queue")
            actual_cmd = (
                # pyre-ignore [16]
                resource["spec"]["tasks"][0]["template"]
                .spec.containers[0]
                .command
            )
            expected_cmd = ["main", "--output-path", "", "--app-id", unique_app_name]
            self.assertEqual(expected_cmd, actual_cmd)

    def test_retry_policy_not_set(self) -> None:
        app = _test_app()
        resource = app_to_resource(app, "test_queue")
        self.assertListEqual(
            [
                {"event": "PodEvicted", "action": "RestartJob"},
                {"event": "PodFailed", "action": "RestartJob"},
            ],
            # pyre-ignore [16]
            resource["spec"]["tasks"][0]["policies"],
        )
        for role in app.roles:
            role.max_retries = 0
        resource = app_to_resource(app, "test_queue")
        self.assertFalse("policies" in resource["spec"]["tasks"][0])
        self.assertFalse("maxRetry" in resource["spec"]["tasks"][0])

    def test_role_to_pod(self) -> None:
        from kubernetes.client.models import (
            V1Pod,
            V1PodSpec,
            V1Container,
            V1EnvVar,
            V1ResourceRequirements,
            V1ContainerPort,
            V1ObjectMeta,
        )

        app = _test_app()
        pod = role_to_pod("name", app.roles[0])

        requests = {
            "cpu": "2000m",
            "memory": "3000M",
            "nvidia.com/gpu": "4",
        }
        resources = V1ResourceRequirements(
            limits=requests,
            requests=requests,
        )
        container = V1Container(
            command=[
                "main",
                "--output-path",
                specs.macros.img_root,
                "--app-id",
                specs.macros.app_id,
            ],
            image="pytorch/torchx:latest",
            name="name",
            env=[V1EnvVar(name="FOO", value="bar")],
            resources=resources,
            ports=[V1ContainerPort(name="foo", container_port=1234)],
        )
        want = V1Pod(
            spec=V1PodSpec(
                containers=[container],
                restart_policy="Never",
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

    def test_validate(self) -> None:
        scheduler = create_scheduler("test")
        app = _test_app()
        scheduler._validate(app, "kubernetes")

    def test_cleanup_str(self) -> None:
        self.assertEqual("abcd123", cleanup_str("abcd123"))
        self.assertEqual("abcd123", cleanup_str("-/_a/b/CD!123!"))
        self.assertEqual("a-bcd123", cleanup_str("-a-bcd123"))
        self.assertEqual("", cleanup_str("!!!"))

    def test_submit_dryrun(self) -> None:
        scheduler = create_scheduler("test")
        app = _test_app()
        cfg = {"queue": "testqueue"}
        with patch(
            "torchx.schedulers.kubernetes_scheduler.make_unique"
        ) as make_unique_ctx:
            make_unique_ctx.return_value = "app-name-42"
            info = scheduler._submit_dryrun(app, cfg)

        resource = str(info.request)

        self.assertEqual(
            resource,
            f"""apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: app-name-42
spec:
  maxRetry: 3
  plugins:
    env: []
    svc: []
  queue: testqueue
  schedulerName: volcano
  tasks:
  - maxRetry: 3
    name: trainer-0
    policies:
    - action: RestartJob
      event: PodEvicted
    - action: RestartJob
      event: PodFailed
    replicas: 1
    template:
      metadata:
        annotations:
          sidecar.istio.io/inject: 'false'
        labels:
          torchx.pytorch.org/app-name: test
          torchx.pytorch.org/replica-id: '0'
          torchx.pytorch.org/role-index: '0'
          torchx.pytorch.org/role-name: trainer
          torchx.pytorch.org/version: {torchx.__version__}
      spec:
        containers:
        - command:
          - main
          - --output-path
          - ''
          - --app-id
          - app-name-42
          env:
          - name: FOO
            value: bar
          image: pytorch/torchx:latest
          name: trainer-0
          ports:
          - containerPort: 1234
            name: foo
          resources:
            limits:
              cpu: 2000m
              memory: 3000M
              nvidia.com/gpu: '4'
            requests:
              cpu: 2000m
              memory: 3000M
              nvidia.com/gpu: '4'
        restartPolicy: Never
""",
        )

    @patch("kubernetes.client.CustomObjectsApi.create_namespaced_custom_object")
    def test_submit(self, create_namespaced_custom_object: MagicMock) -> None:
        create_namespaced_custom_object.return_value = {
            "metadata": {"name": "testid"},
        }
        scheduler = create_scheduler("test")
        app = _test_app()
        cfg = {
            "namespace": "testnamespace",
            "queue": "testqueue",
        }

        info = scheduler._submit_dryrun(app, cfg)
        id = scheduler.schedule(info)
        self.assertEqual(id, "testnamespace:testid")
        call = create_namespaced_custom_object.call_args
        args, kwargs = call
        self.assertEqual(kwargs["group"], "batch.volcano.sh")
        self.assertEqual(kwargs["version"], "v1alpha1")
        self.assertEqual(kwargs["namespace"], "testnamespace")
        self.assertEqual(kwargs["plural"], "jobs")
        self.assertEqual(kwargs["body"], info.request.resource)

    @patch("kubernetes.client.CustomObjectsApi.create_namespaced_custom_object")
    def test_submit_job_name_conflict(
        self, create_namespaced_custom_object: MagicMock
    ) -> None:
        from kubernetes.client.rest import ApiException

        api_exc = ApiException(status=409, reason="Conflict")
        api_exc.body = "{'details':{'name': 'test_job'}}"
        create_namespaced_custom_object.side_effect = api_exc

        scheduler = create_scheduler("test")
        app = _test_app()
        cfg = {
            "namespace": "testnamespace",
            "queue": "testqueue",
        }
        info = scheduler._submit_dryrun(app, cfg)
        with self.assertRaises(ValueError):
            scheduler.schedule(info)

    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object_status")
    def test_describe(self, get_namespaced_custom_object_status: MagicMock) -> None:
        get_namespaced_custom_object_status.return_value = {
            "status": {
                "state": {"phase": "Completed"},
                "succeeded": 1,
                "taskStatusCount": {"echo-0": {"phase": {"Succeeded": 1}}},
            }
        }
        app_id = "testnamespace:testid"
        scheduler = create_scheduler("test")
        info = scheduler.describe(app_id)
        call = get_namespaced_custom_object_status.call_args
        args, kwargs = call
        self.assertEqual(
            kwargs,
            {
                "group": "batch.volcano.sh",
                "version": "v1alpha1",
                "namespace": "testnamespace",
                "plural": "jobs",
                "name": "testid",
            },
        )
        self.assertEqual(
            info,
            DescribeAppResponse(
                app_id=app_id,
                state=specs.AppState.SUCCEEDED,
                roles_statuses=[
                    specs.RoleStatus(
                        "echo",
                        [
                            specs.ReplicaStatus(
                                id=0,
                                role="echo",
                                state=specs.AppState.SUCCEEDED,
                                hostname="",
                            )
                        ],
                    ),
                ],
                roles=[
                    specs.Role(name="echo", image="", num_replicas=1),
                ],
            ),
        )

    @patch("kubernetes.client.CustomObjectsApi.get_namespaced_custom_object_status")
    def test_describe_unknown(
        self, get_namespaced_custom_object_status: MagicMock
    ) -> None:
        get_namespaced_custom_object_status.return_value = {}
        app_id = "testnamespace:testid"
        scheduler = create_scheduler("test")
        info = scheduler.describe(app_id)
        call = get_namespaced_custom_object_status.call_args
        args, kwargs = call
        self.assertEqual(
            kwargs,
            {
                "group": "batch.volcano.sh",
                "version": "v1alpha1",
                "namespace": "testnamespace",
                "plural": "jobs",
                "name": "testid",
            },
        )
        self.assertEqual(
            info,
            DescribeAppResponse(
                app_id=app_id,
                state=specs.AppState.UNKNOWN,
            ),
        )

    def test_runopts(self) -> None:
        scheduler = kubernetes_scheduler.create_scheduler("foo")
        runopts = scheduler.run_opts()
        self.assertEqual(set(runopts._opts.keys()), {"queue", "namespace"})

    @patch("kubernetes.client.CustomObjectsApi.delete_namespaced_custom_object")
    def test_cancel_existing(self, delete_namespaced_custom_object: MagicMock) -> None:
        scheduler = create_scheduler("test")
        scheduler._cancel_existing("testnamespace:testjob")
        call = delete_namespaced_custom_object.call_args
        args, kwargs = call
        self.assertEqual(
            kwargs,
            {
                "group": "batch.volcano.sh",
                "version": "v1alpha1",
                "namespace": "testnamespace",
                "plural": "jobs",
                "name": "testjob",
            },
        )

    @patch("kubernetes.client.CoreV1Api.read_namespaced_pod_log")
    def test_log_iter(self, read_namespaced_pod_log: MagicMock) -> None:
        scheduler = create_scheduler("test")
        read_namespaced_pod_log.return_value = "foo reg\nfoo\nbar reg\n"
        lines = scheduler.log_iter(
            app_id="testnamespace:testjob",
            role_name="role",
            k=1,
            regex="reg",
            since=datetime.now(),
        )
        self.assertEqual(
            list(lines),
            [
                "foo reg",
                "bar reg",
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
                "name": "testjob-role-1-0",
                "timestamps": True,
            },
        )


class KubernetesSchedulerNoImportTest(unittest.TestCase):
    """
    KubernetesSchedulerNoImportTest tests the kubernetes scheduler behavior when
    Kubernetes is not available.
    """

    def setUp(self) -> None:
        # make all kubernetes modules unable to be imported
        for mod in list(sys.modules.keys()) + ["kubernetes"]:
            if mod.startswith("kubernetes"):
                sys.modules[mod] = None  # pyre-ignore

        # reload to ensure kubernetes_scheduler doesn't depend on them at import
        # time
        importlib.reload(kubernetes_scheduler)
        importlib.reload(schedulers)

    def tearDown(self) -> None:
        # reset all kubernetes modules we patched
        for mod in list(sys.modules.keys()):
            if mod.startswith("kubernetes"):
                del sys.modules[mod]
        # reimport kubernetes_scheduler to get to a clean state
        importlib.reload(kubernetes_scheduler)

    def test_runopts(self) -> None:
        scheduler = kubernetes_scheduler.create_scheduler("foo")
        self.assertIsNotNone(scheduler.run_opts())

    def test_describe(self) -> None:
        scheduler = kubernetes_scheduler.create_scheduler("foo")
        with self.assertRaises(ModuleNotFoundError):
            scheduler.describe("foo:bar")

    def test_dryrun(self) -> None:
        scheduler = kubernetes_scheduler.create_scheduler("foo")
        app = _test_app()
        cfg = {
            "namespace": "testnamespace",
            "queue": "testqueue",
        }

        with self.assertRaises(ModuleNotFoundError):
            scheduler._submit_dryrun(app, cfg)
