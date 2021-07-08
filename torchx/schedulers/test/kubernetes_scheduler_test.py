# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from unittest.mock import patch, MagicMock

from kubernetes.client.models import (
    V1Pod,
    V1PodSpec,
    V1Container,
    V1EnvVar,
    V1ResourceRequirements,
    V1ContainerPort,
)
from torchx import specs
from torchx.schedulers.api import DescribeAppResponse
from torchx.schedulers.kubernetes_scheduler import (
    create_scheduler,
    KubernetesScheduler,
    role_to_pod,
)


class KubernetesSchedulerTest(unittest.TestCase):
    def test_create_scheduler(self) -> None:
        scheduler = create_scheduler("foo")
        self.assertIsInstance(scheduler, KubernetesScheduler)

    def _test_app(self) -> specs.AppDef:
        trainer_role = specs.Role(
            name="trainer",
            image="pytorch/torchx:latest",
            entrypoint="main",
            args=["--output-path", specs.macros.img_root],
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

    def test_role_to_pod(self) -> None:
        app = self._test_app()
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
            command=["main", "--output-path", specs.macros.img_root],
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
        )

        self.assertEqual(
            pod,
            want,
        )

    def test_validate(self) -> None:
        scheduler = create_scheduler("test")
        app = self._test_app()
        scheduler._validate(app, "kubernetes")

    def test_submit_dryrun(self) -> None:
        scheduler = create_scheduler("test")
        app = self._test_app()
        cfg = specs.RunConfig()
        cfg.set("queue", "testqueue")
        info = scheduler._submit_dryrun(app, cfg)

        resource = str(info.request)

        self.assertEqual(
            resource,
            """apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  generateName: test-
spec:
  maxRetry: 3
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
      spec:
        containers:
        - command:
          - main
          - --output-path
          - ''
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
        app = self._test_app()
        cfg = specs.RunConfig()
        cfg.set("namespace", "testnamespace")
        cfg.set("queue", "testqueue")
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
            ),
        )
