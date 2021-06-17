#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import os
import os.path
import shlex
from typing import Dict, List, Optional, Protocol, Tuple, Type, Mapping, Callable

import yaml
from kfp import components, dsl

# @manual=fbsource//third-party/pypi/kfp:kfp
from kfp.components.structures import OutputSpec, ComponentSpec
from kubernetes.client.models import (
    V1ContainerPort,
    V1Volume,
    V1VolumeMount,
    V1EmptyDirVolumeSource,
)
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
        "outputs": [],
    }
    return yaml.dump(spec), role


class ContainerFactory(Protocol):
    """
    ContainerFactory is a protocol that represents a function that when called produces a
    kfp.dsl.ContainerOp.
    """

    def __call__(self, *args: object, **kwargs: object) -> dsl.ContainerOp:
        ...


class KFPContainerFactory(ContainerFactory, Protocol):
    """
    KFPContainerFactory is a ContainerFactory that also has some KFP metadata
    attached to it.
    """

    component_spec: ComponentSpec


METADATA_FILE = "/tmp/outputs/mlpipeline-ui-metadata/data.json"


def component_from_app(
    app: api.AppDef, ui_metadata: Optional[Mapping[str, object]] = None
) -> ContainerFactory:
    """
    component_from_app takes in a TorchX component/AppDef and returns a KFP
    ContainerOp factory. This is equivalent to the
    `kfp.components.load_component_from_*
    <https://kubeflow-pipelines.readthedocs.io/en/stable/source/kfp.components.html#kfp.components.load_component_from_text>`_
    methods.

    Args:
        app: The AppDef to generate a KFP container factory for.
        ui_metadata: KFP UI Metadata to output so you can have model results show
            up in the UI. See
            https://www.kubeflow.org/docs/components/pipelines/sdk/output-viewer/
            for more info on the format.

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
    component_factory: KFPContainerFactory = components.load_component_from_text(spec)

    if ui_metadata is not None:
        # pyre-fixme[16]: `ComponentSpec` has no attribute `outputs`
        component_factory.component_spec.outputs.append(
            OutputSpec(
                name="mlpipeline-ui-metadata",
                type="MLPipeline UI Metadata",
                description="ui metadata",
            )
        )

    def factory_wrapper(*args: object, **kwargs: object) -> dsl.ContainerOp:
        c = component_factory(*args, **kwargs)
        container = c.container

        if ui_metadata is not None:
            # We generate the UI metadata from the sidecar so we need to make
            # both the container and the sidecar share the same tmp directory so
            # the outputs appear in the original container.
            c.add_volume(V1Volume(name="tmp", empty_dir=V1EmptyDirVolumeSource()))
            container.add_volume_mount(
                V1VolumeMount(
                    name="tmp",
                    mount_path="/tmp/",
                )
            )
            c.output_artifact_paths["mlpipeline-ui-metadata"] = METADATA_FILE
            c.add_sidecar(_ui_metadata_sidecar(ui_metadata))

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


def _ui_metadata_sidecar(
    ui_metadata: Mapping[str, object], image: str = "alpine"
) -> dsl.Sidecar:
    shell_encoded = shlex.quote(json.dumps(ui_metadata))
    dirname = os.path.dirname(METADATA_FILE)
    return dsl.Sidecar(
        name="ui-metadata-sidecar",
        image=image,
        command=[
            "sh",
            "-c",
            f"mkdir -p {dirname}; echo {shell_encoded} > {METADATA_FILE}",
        ],
        mirror_volume_mounts=True,
    )
