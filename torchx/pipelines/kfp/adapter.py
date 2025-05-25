#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
This module contains adapters for converting TorchX components to
Kubeflow Pipeline (KFP) v2 components.
"""

import json
from typing import Any, Dict, Optional, Tuple

import yaml
from kfp import dsl
from kfp.dsl import ContainerSpec, OutputPath, PipelineTask

import torchx.specs as api
from torchx.schedulers import kubernetes_scheduler


# Metadata Template for TorchX components
UI_METADATA_TEMPLATE = """
import json
metadata = {metadata_json}
with open("{output_path}", "w") as f:
    json.dump(metadata, f)
"""


def component_from_app(
    app: api.AppDef,
    ui_metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    component_from_app creates a KFP v2 component from a TorchX AppDef.
    
    In KFP v2, we use container components for single-container apps.
    For multi-role apps, we use the first role as the primary container.

    Args:
        app: The torchx AppDef to adapt.
        ui_metadata: optional UI metadata to attach to the component.

    Returns:
        A KFP v2 component function.

    Note:
        KFP v2 uses a different component structure than v1. This function
        returns a component that can be used within a pipeline function.

    Example:
        >>> from torchx import specs
        >>> from torchx.pipelines.kfp.adapter import component_from_app
        >>> from kfp import dsl
        >>> 
        >>> app_def = specs.AppDef(
        ...     name="trainer",
        ...     roles=[specs.Role(
        ...         name="trainer", 
        ...         image="pytorch/pytorch:latest",
        ...         entrypoint="python",
        ...         args=["-m", "train", "--epochs", "10"],
        ...         env={"CUDA_VISIBLE_DEVICES": "0"},
        ...         resource=specs.Resource(cpu=2, memMB=8192, gpu=1)
        ...     )],
        ... )
        >>> trainer_component = component_from_app(app_def)
        >>> 
        >>> @dsl.pipeline(name="training-pipeline")
        >>> def my_pipeline():
        ...     trainer_task = container_from_app(app_def)
        ...     trainer_task.set_display_name("Model Training")
    """
    if len(app.roles) > 1:
        raise ValueError(
            f"KFP adapter does not support apps with more than one role. "
            f"AppDef has roles: {[r.name for r in app.roles]}"
        )
    
    role = app.roles[0]
    
    @dsl.container_component
    def torchx_component(mlpipeline_ui_metadata: OutputPath(str) = None) -> ContainerSpec:
        """KFP v2 wrapper for TorchX component."""
        # Basic container spec
        container_spec_dict = {
            "image": role.image,
        }
        
        # Build command and args
        command = []
        if role.entrypoint:
            command.append(role.entrypoint)
        if role.args:
            command.extend(role.args)
        
        # Set command or args
        if role.entrypoint and role.args:
            # If both entrypoint and args exist, use command for full command line
            container_spec_dict["command"] = command
        elif role.entrypoint:
            # If only entrypoint exists, use it as command
            container_spec_dict["command"] = [role.entrypoint]
        elif role.args:
            # If only args exist, use them as args
            container_spec_dict["args"] = list(role.args)
        
        # Handle UI metadata if provided
        if ui_metadata and mlpipeline_ui_metadata:
            metadata_json = json.dumps(ui_metadata)
            metadata_cmd = f'echo \'{metadata_json}\' > {mlpipeline_ui_metadata}'
            
            # If there's an existing command, wrap it with metadata writing
            if "command" in container_spec_dict:
                existing_cmd = ' '.join(container_spec_dict["command"])
                container_spec_dict["command"] = ["sh", "-c", f"{metadata_cmd} && {existing_cmd}"]
            else:
                container_spec_dict["command"] = ["sh", "-c", metadata_cmd]
        
        return ContainerSpec(**container_spec_dict)
    
    # Set component metadata
    torchx_component._component_human_name = f"{app.name}-{role.name}"
    torchx_component._component_description = f"KFP v2 wrapper for TorchX component {app.name}, role {role.name}"
    
    # Store role resource info as a component attribute so container_from_app can use it
    torchx_component._torchx_role = role
    
    return torchx_component


# Alias for clarity - matches the naming in adapter_v23.py
component_from_app_def = component_from_app


def container_from_app(
    app: api.AppDef,
    *args: Any,
    ui_metadata: Optional[Dict[str, Any]] = None,
    display_name: Optional[str] = None,
    retry_policy: Optional[Dict[str, Any]] = None,
    enable_caching: bool = True,
    accelerator_type: Optional[str] = None,
    **kwargs: Any,
) -> PipelineTask:
    """
    container_from_app transforms the app into a KFP v2 component and returns a
    corresponding PipelineTask instance when called within a pipeline.

    Args:
        app: The torchx AppDef to adapt.
        ui_metadata: optional UI metadata to attach to the component.
        display_name: optional display name for the task in the KFP UI.
        retry_policy: optional retry configuration dict with 'max_retry_count' and/or 'backoff_duration' keys.
        enable_caching: whether to enable caching for this task (default: True).
        accelerator_type: optional accelerator type (e.g., 'nvidia-tesla-v100', 'nvidia-tesla-k80').
        *args: positional arguments passed to the component.
        **kwargs: keyword arguments passed to the component.

    Returns:
        A configured PipelineTask instance.

    See component_from_app for description on the arguments. Any unspecified
    arguments are passed through to the component.

    Example:
        >>> import kfp
        >>> from kfp import dsl
        >>> from torchx import specs
        >>> from torchx.pipelines.kfp.adapter import container_from_app
        >>> 
        >>> app_def = specs.AppDef(
        ...     name="trainer",
        ...     roles=[specs.Role(
        ...         name="trainer", 
        ...         image="pytorch/pytorch:latest",
        ...         entrypoint="python",
        ...         args=["train.py"],
        ...         resource=specs.Resource(cpu=4, memMB=16384, gpu=1)
        ...     )],
        ... )
        >>> 
        >>> @dsl.pipeline(name="ml-pipeline")
        >>> def pipeline():
        ...     # Create a training task
        ...     trainer = container_from_app(
        ...         app_def, 
        ...         display_name="PyTorch Training",
        ...         retry_policy={'max_retry_count': 3},
        ...         accelerator_type='nvidia-tesla-v100'
        ...     )
        ...     trainer.set_env_variable("WANDB_PROJECT", "my-project")
        ...     
        ...     # Create another task that depends on trainer
        ...     evaluator = container_from_app(
        ...         app_def,
        ...         display_name="Model Evaluation"
        ...     )
        ...     evaluator.after(trainer)
    """
    component = component_from_app(app, ui_metadata)
    # Call the component function to create a PipelineTask
    task = component(*args, **kwargs)
    
    # Apply resource constraints and environment variables from the role
    if hasattr(component, '_torchx_role'):
        role = component._torchx_role
        
        # Set resources
        if role.resource.cpu > 0:
            task.set_cpu_request(str(int(role.resource.cpu)))
            task.set_cpu_limit(str(int(role.resource.cpu)))
        if role.resource.memMB > 0:
            task.set_memory_request(f"{role.resource.memMB}M")
            task.set_memory_limit(f"{role.resource.memMB}M")
        if role.resource.gpu > 0:
            # Use the newer set_accelerator_limit API (set_gpu_limit is deprecated)
            # Check for accelerator type in metadata or use provided one
            acc_type = accelerator_type
            if not acc_type and app.metadata:
                acc_type = app.metadata.get('accelerator_type')
            if not acc_type:
                acc_type = 'nvidia-tesla-k80'  # Default GPU type
            
            task.set_accelerator_type(acc_type)
            task.set_accelerator_limit(str(int(role.resource.gpu)))
        
        # Set environment variables
        if role.env:
            for name, value in role.env.items():
                task.set_env_variable(name=name, value=str(value))
    
    # Apply additional configurations
    if display_name:
        task.set_display_name(display_name)
    
    if retry_policy:
        retry_args = {}
        if 'max_retry_count' in retry_policy:
            retry_args['num_retries'] = retry_policy['max_retry_count']
        if 'backoff_duration' in retry_policy:
            retry_args['backoff_duration'] = retry_policy['backoff_duration']
        if 'backoff_factor' in retry_policy:
            retry_args['backoff_factor'] = retry_policy['backoff_factor']
        if 'backoff_max_duration' in retry_policy:
            retry_args['backoff_max_duration'] = retry_policy['backoff_max_duration']
        if retry_args:
            task.set_retry(**retry_args)
    
    # Set caching options
    task.set_caching_options(enable_caching=enable_caching)
    
    return task


def resource_from_app(
    app: api.AppDef,
    queue: str,
    service_account: Optional[str] = None,
    priority_class: Optional[str] = None,
) -> PipelineTask:
    """
    resource_from_app generates a KFP v2 component that creates Kubernetes
    resources using the Volcano job scheduler for distributed apps.

    Args:
        app: The torchx AppDef to adapt.
        queue: the Volcano queue to schedule the job in.
        service_account: optional service account to use.
        priority_class: optional priority class name.

    Note: In KFP v2, direct Kubernetes resource manipulation requires
    the kfp-kubernetes extension. This function provides a basic
    implementation using kubectl.

    >>> import kfp
    >>> from torchx import specs
    >>> from torchx.pipelines.kfp.adapter import resource_from_app
    >>> app_def = specs.AppDef(
    ...     name="trainer",
    ...     roles=[specs.Role("trainer", image="foo:latest", num_replicas=3)],
    ... )
    >>> @dsl.pipeline
    >>> def pipeline():
    ...     trainer = resource_from_app(app_def, queue="test")
    ...     print(trainer)
    """
    @dsl.container_component
    def volcano_job_component() -> ContainerSpec:
        """Creates a Volcano job via kubectl."""
        resource = kubernetes_scheduler.app_to_resource(
            app, queue, service_account, priority_class
        )
        
        # Serialize the resource to YAML
        resource_yaml = yaml.dump(resource, default_flow_style=False)
        
        # Use kubectl to create the resource
        return ContainerSpec(
            image="bitnami/kubectl:latest",
            command=["sh", "-c"],
            args=[f"echo '{resource_yaml}' | kubectl apply -f -"],
        )
    
    volcano_job_component._component_human_name = f"{app.name}-volcano-job"
    volcano_job_component._component_description = f"Creates Volcano job for {app.name}"
    
    return volcano_job_component()


# Backwards compatibility - map old function names to new ones
def component_spec_from_app(app: api.AppDef) -> Tuple[str, api.Role]:
    """
    DEPRECATED: This function is maintained for backwards compatibility.
    Use component_from_app instead for KFP v2.
    """
    import warnings
    warnings.warn(
        "component_spec_from_app is deprecated. Use component_from_app for KFP v2.",
        DeprecationWarning,
        stacklevel=2
    )
    
    if len(app.roles) != 1:
        raise ValueError(
            f"Distributed apps are only supported via resource_from_app. "
            f"{app.name} has roles: {[r.name for r in app.roles]}"
        )
    
    return app.name, app.roles[0]


def component_spec_from_role(name: str, role: api.Role) -> Dict[str, Any]:
    """
    DEPRECATED: Use component_from_app for KFP v2.
    """
    import warnings
    warnings.warn(
        "component_spec_from_role is deprecated. Use component_from_app for KFP v2.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Return a minimal spec for backwards compatibility
    return {
        "name": f"{name}-{role.name}",
        "description": f"DEPRECATED: {name} {role.name}",
    }
