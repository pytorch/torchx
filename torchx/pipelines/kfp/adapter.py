#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from typing import Callable, Dict, List, Optional, Protocol, Tuple, Type

import yaml
from kfp import components, dsl
from kubernetes.client.models import V1ContainerPort
from torchx.runtime.component import Component, is_optional
from torchx.specs import api

from .version import __version__ as __version__  # noqa F401


TORCHX_CONTAINER_ENV: str = "TORCHX_CONTAINER"
TORCHX_CONTAINER: str = os.getenv(
    TORCHX_CONTAINER_ENV,
    "pytorch/torchx:latest",
)
TORCHX_ENTRY_POINT = "torchx/container/main.py"
PYTHON_COMMAND = "python3"


# pyre-fixme[24]: Generic type `Component` expects 3 type parameters.
def component_spec(c: Type[Component], image: Optional[str] = None) -> str:
    """
    deprecated: do not use
    """

    assert issubclass(c, Component), f"{c} must be a subclass of Component"
    inputs = []
    outputs = []
    qualname = f"{c.__module__}.{c.__qualname__}"

    command: List[object] = [
        PYTHON_COMMAND,
        TORCHX_ENTRY_POINT,
        qualname,
    ]

    Config, Inputs, Outputs = c._get_args()
    for arg in (Config, Inputs, Outputs):
        for fieldname, fieldtype in arg.__annotations__.items():
            inp = {"name": fieldname, "type": "String"}
            if is_optional(fieldtype):
                inp["default"] = "null"
            inputs.append(inp)
            command += [
                f"--{fieldname}",
                {"inputValue": fieldname},
            ]
            if arg == Outputs:
                outputs.append(copy.deepcopy(inp))
                command += [
                    f"--output-path-{fieldname}",
                    {"outputPath": fieldname},
                ]

    spec = {
        "name": c.__name__,
        "description": f"KFP wrapper for TorchX component {qualname}. Version: {c.Version}",
        "inputs": inputs,
        "outputs": outputs,
        "implementation": {
            "container": {
                "image": image or TORCHX_CONTAINER,
                "command": command,
            }
        },
    }

    return yaml.dump(spec)


# pyre-fixme[24]: Generic type `Component` expects 3 type parameters.
# pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
def component_op(c: Type[Component], image: Optional[str] = None) -> Callable:
    """
    deprecated: do not use
    """

    spec = component_spec(c, image=image)
    return components.load_component_from_text(spec)


class TorchXComponent:
    """
    deprecated: do not use
    """

    # pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
    _factory: Optional[Callable] = None

    def __init_subclass__(
        cls,
        *args: object,
        # pyre-fixme[24]: Generic type `Component` expects 3 type parameters.
        component: Optional[Type[Component]] = None,
        image: Optional[str] = None,
        **kwargs: object,
    ) -> None:
        assert component and issubclass(
            component, Component
        ), f"must specify component, got {component}"
        cls._factory = component_op(component, image=image)

        super().__init_subclass__(*args, **kwargs)

    def __new__(cls, *args: object, **kwargs: object) -> "TorchXComponent":
        factory = cls._factory
        assert factory, "must have component"
        return factory(*args, **kwargs)

    # These methods are never run since we override the __new__ method but it gives us type checking.

    @property
    def outputs(self) -> Dict[str, dsl.PipelineParam]:
        ...

    @property
    def output(self) -> dsl.PipelineParam:
        ...


def component_spec_from_app(app: api.AppDef) -> Tuple[str, api.Container]:
    assert len(app.roles) == 1, f"KFP adapter only support one role, got {app.roles}"

    role = app.roles[0]
    assert (
        role.num_replicas == 1
    ), f"KFP adapter only supports one replica, got {app.num_replicas}"
    assert role.container != api.NULL_CONTAINER, "missing container for KFP"

    container = role.container
    assert container.base_image is None, "KFP adapter does not support base_image"

    command = [role.entrypoint, *role.args]

    spec = {
        "name": f"{app.name}-{role.name}",
        "description": f"KFP wrapper for TorchX component {app.name}, role {role.name}",
        "implementation": {
            "container": {
                "image": container.image,
                "command": command,
                "env": role.env,
            }
        },
    }
    return yaml.dump(spec), container


class ContainerFactory(Protocol):
    def __call__(self, *args: object, **kwargs: object) -> dsl.ContainerOp:
        ...


def component_from_app(app: api.AppDef) -> ContainerFactory:
    container_spec: api.Container
    spec, container_spec = component_spec_from_app(app)
    resources: api.Resource = container_spec.resources
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
        if (gpu := resources.gpu) >= 0:
            container.set_gpu_limit(str(gpu))

        for name, port in container_spec.port_map.items():
            container.add_port(
                V1ContainerPort(
                    name=name,
                    container_port=port,
                ),
            )

        return c

    return factory_wrapper
