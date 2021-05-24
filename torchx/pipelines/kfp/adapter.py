#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from typing import Callable, Dict, List, Optional, Type

import yaml
from kfp import components, dsl
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
    spec = component_spec(c, image=image)
    return components.load_component_from_text(spec)


class TorchXComponent:
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


def component_spec_from_app(app: api.Application) -> str:
    assert len(app.roles) == 1, f"KFP adapter only support one role, got {app.roles}"

    role = app.roles[0]
    assert (
        role.num_replicas == 1
    ), f"KFP adapter only supports one replica, got {app.num_replicas}"
    assert role.container != api.NULL_CONTAINER, "missing container for KFP"

    container = role.container
    assert container.base_image is None, "KFP adapter does not support base_image"
    assert (
        container.resources == api.NULL_RESOURCE
    ), "KFP adapter requires you to specify resources in the pipeline"
    assert len(container.port_map) == 0, "KFP adapter does not support port_map"

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
    return yaml.dump(spec)


# pyre-fixme[24]: Generic type `Callable` expects 2 type parameters.
def component_from_app(app: api.Application) -> Callable:
    spec = component_spec_from_app(app)
    return components.load_component_from_text(spec)
