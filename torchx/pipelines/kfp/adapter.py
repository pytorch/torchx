#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Protocol, Tuple

import yaml
from kfp import components, dsl
from kubernetes.client.models import V1ContainerPort
from torchx.specs import api

from .version import __version__ as __version__  # noqa F401


def component_spec_from_app(app: api.AppDef) -> Tuple[str, api.Role]:
    """
    component_spec_from_app takes in a TorchX component and generates the yaml
    spec for it. Notably this doesn't apply resources or port_maps since those
    must be applied at runtime which is why it returns the role spec as well.

    >>> from torchx import specs
    >>> from torchx.pipelines.kfp.adapter import component_spec_from_app
    >>> app_def = specs.AppDef(
    ...     name="trainer",
    ...     roles=[specs.Role("trainer", image="foo:latest")],
    ... )
    >>> component_spec_from_app(app_def)
    ('description: ...', Role(...))
    """
    assert len(app.roles) == 1, f"KFP adapter only support one role, got {app.roles}"

    role = app.roles[0]
    assert (
        role.num_replicas == 1
    ), f"KFP adapter only supports one replica, got {app.num_replicas}"

    assert role.base_image is None, "KFP adapter does not support base_image"

    command = [role.entrypoint, *role.args]

    spec = {
        "name": f"{app.name}-{role.name}",
        "description": f"KFP wrapper for TorchX component {app.name}, role {role.name}",
        "implementation": {
            "container": {
                "image": role.image,
                "command": command,
                "env": role.env,
            }
        },
    }
    return yaml.dump(spec), role


class ContainerFactory(Protocol):
    """
    ContainerFactory is a protocol that represents a function that when called produces a
    kfp.dsl.ContainerOp.
    """

    def __call__(self, *args: object, **kwargs: object) -> dsl.ContainerOp:
        ...


def component_from_app(app: api.AppDef) -> ContainerFactory:
    """
    component_from_app takes in a TorchX component/AppDef and returns a KFP
    ContainerOp factory. This is equivalent to the
    `kfp.components.load_component_from_*
    <https://kubeflow-pipelines.readthedocs.io/en/stable/source/kfp.components.html#kfp.components.load_component_from_text>`_
    methods.

    >>> from torchx import specs
    >>> from torchx.pipelines.kfp.adapter import component_from_app
    >>> app_def = specs.AppDef(
    ...     name="trainer",
    ...     roles=[specs.Role("trainer", image="foo:latest")],
    ... )
    >>> component_from_app(app_def)
    <function component_from_app...>
    """

    role_spec: api.Role
    spec, role_spec = component_spec_from_app(app)
    resources: api.Resource = role_spec.resource
    assert (
        len(resources.capabilities) == 0
    ), f"KFP doesn't support capabilities, got {resources.capabilities}"
    component_factory: ContainerFactory = components.load_component_from_text(spec)

    def factory_wrapper(*args: object, **kwargs: object) -> dsl.ContainerOp:
        c = component_factory(*args, **kwargs)
        container = c.container

        if (cpu := resources.cpu) >= 0:
            cpu_str = f"{int(cpu*1000)}m"
            container.set_cpu_request(cpu_str)
            container.set_cpu_limit(cpu_str)
        if (mem := resources.memMB) >= 0:
            mem_str = f"{int(mem)}M"
            container.set_memory_request(mem_str)
            container.set_memory_limit(mem_str)
        if (gpu := resources.gpu) > 0:
            container.set_gpu_limit(str(gpu))

        for name, port in role_spec.port_map.items():
            container.add_port(
                V1ContainerPort(
                    name=name,
                    container_port=port,
                ),
            )

        return c

    return factory_wrapper
