#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""

Initial support for Azure Batch scheduler.

This scheduler is in prototype stage and may change without notice.

Overview
========
https://learn.microsoft.com/en-us/azure/batch/


Prerequisites
==============

You'll need to setup:
- Azure Batch Account+credentials.
- Batch Pool, that:
    - has enough resources, or
    - supports autoscaling (https://learn.microsoft.com/en-us/azure/batch/batch-automatic-scaling)
    - VMs that support container workloads (https://learn.microsoft.com/en-us/azure/batch/batch-docker-container-workloads)

Configuration
=============
Use .torchxconfig to configure the Batch Account information

.. code-block:: ini
[scheduler:azure_batch]
batch_account_name = <ACCOUNT NAME>
batch_account_key = "<ACCOUNT_KEY>"
batch_account_url: <ACCOUNT_URL>
.. code-block:: ini

Although workspace patching support is WIP, the scheduler can be configured to use existing images by configuring Container Registry.

"""

from dataclasses import dataclass
from typing import Dict, List, Optional, TypedDict
import yaml
from torchx.schedulers.api import (
    AppDryRunInfo,
    DescribeAppResponse,
    ListAppResponse,
    Scheduler
)
from torchx.schedulers.ids import make_unique

from torchx.specs.api import (
    AppDef,
    AppState,
    macros,
    runopts,
    RetryPolicy,
    Role,
)

from azure.batch import BatchServiceClient

import azure.batch.models as batchmodels
from azure.batch import BatchServiceClient
from azure.batch.batch_auth import SharedKeyCredentials
import azure.batch.models as batchmodels



@dataclass
class AzureBatchJob:
    job_id: str
    pool_id: str
    job: batchmodels.JobAddParameter
    tasks: List[batchmodels.TaskAddParameter]

    def __str__(self) -> str:
        return yaml.dump({"job": self.job, "tasks": self.tasks})

    def __repr__(self) -> str:
        return str(self)


class AzureBatchOpts(TypedDict, total=False):
    batch_container_registry: Optional[str]
    batch_container_username: Optional[str]
    batch_container_password: Optional[str]
    batch_pool_id: Optional[str]

    
JOB_STATE: Dict[str, AppState] = {
    # TODO: instead of state to state mapping, use Job transition states (prev+current) to capture status
    batchmodels.JobState.active: AppState.RUNNING,
    batchmodels.JobState.completed: AppState.SUCCEEDED,
    batchmodels.JobState.deleting: AppState.UNKNOWN,
    batchmodels.JobState.disabled: AppState.UNKNOWN,
    batchmodels.JobState.disabling: AppState.UNKNOWN,
    batchmodels.JobState.enabling: AppState.PENDING,
    batchmodels.JobState.terminating: AppState.UNKNOWN,
}

def _as_metadata_model(name:str, value:str) -> batchmodels.MetadataItem:
    # pyre-ignore[16]
    return batchmodels.MetadataItem(name=name, value=value)

class AzureBatchScheduler(Scheduler[AzureBatchOpts]):
    """
    AzureBatchScheduler is a TorchX scheduling interface to Azure Batch.

    .. code-block:: bash

        $ pip install torchx
        $ torchx run --scheduler azure_batch --scheduler_args image_repo=ghcr.io/pytorch/torchx --image alpine:latest --msg hello
        azure_batch://torchx_user/1234
        $ torchx describe azure_batch://torchx_user/1234
        ...

    **Config Options**

    .. runopts::
        class: torchx.schedulers.azure_batch_scheduler.create_scheduler


    **Compatibility**

    .. compatibility::
        type: scheduler
        features:
            cancel: true
            logs: false
            distributed: true
            describe: true
            workspaces: false
            mounts: false
            elasticity: false
    """

    def __init__(
        self,
        session_name: str,
        batch_account_name: Optional[str],
        batch_account_key: Optional[str],
        batch_account_url: Optional[str],
    ) -> None:
        super().__init__("azure_batch", session_name)
        self.batch_account_name = batch_account_name
        self.batch_account_key = batch_account_key
        self.batch_account_url = batch_account_url

    def _build_client(self) -> BatchServiceClient:
        credentials = SharedKeyCredentials(
            self.batch_account_name,
            self.batch_account_key)

        batch_service_client = BatchServiceClient(
            credentials,
            batch_url=self.batch_account_url)
        
        return batch_service_client
    

    def schedule(self, dryrun_info: AppDryRunInfo[AzureBatchJob]) -> str:
        req = dryrun_info.request

        batch_service_client = self._build_client()
        pool_id = dryrun_info.request.pool_id
        if not batch_service_client.pool.exists(pool_id=pool_id):
            # TODO: add a support for creating autoscaling pool. Currently we expect the pool to exist
            raise ValueError(f"Batch pool {pool_id} does not exist")
        job = req.job
        
        batch_service_client.job.add(job)

        for task in req.tasks:
            batch_service_client.task.add(req.job_id, task)
        return req.job_id

    
    def _role_as_task(self, name: str, app_role: Role, cfg: AzureBatchOpts) -> batchmodels.TaskAddParameter:
        values = macros.Values(
                    img_root="",
                    app_id=name,
                    replica_id="",
                    rank0_env="AZ_BATCH_MASTER_NODE",
                )
        role = values.apply(app_role)
        role_id=f'{name}-{role.name}'
        if role.num_replicas == 1:
            task = batchmodels.TaskAddParameter( # pyre-ignore[16]
                id=role_id,
                command_line = " ".join([role.entrypoint] + role.args) , 
            )
        elif role.num_replicas > 1:
            task = batchmodels.TaskAddParameter( # pyre-ignore[16]
                id=role_id,
                command_line='echo "MPI Primary task started"', # required
            )
            # pyre-ignore[16]
            multi_instance_settings=batchmodels.MultiInstanceSettings( 
                number_of_instances=role.num_replicas,
                coordination_command_line=" ".join([role.entrypoint] + role.args)
            )
            task.multi_instance_settings = multi_instance_settings
        else:
            raise ValueError(f"Expected positive {role.num_replicas} for {role.name} role")


        task.environment_settings = [
            batchmodels.EnvironmentSetting(name="TORCHX_ROLE_NAME", value=str(role.name))] # pyre-ignore[16]

        if role.env:
            task.environment_settings += [
                    batchmodels.EnvironmentSetting(name=name, value=value) # pyre-ignore[16]
                    for name, value in role.env.items()]

        # Scheduler supports only "RetryPolicy.APPLICATION" semantics
        if role.max_retries:
            if role.retry_policy == RetryPolicy.REPLICA:
                raise ValueError("Scheduler supports only RetryPolicy.APPLICATION option")
            # pyre-ignore[16]
            task.constraints = batchmodels.TaskConstraints( 
                max_task_retry_count=role.max_retries)

        if role.image:
            # pyre-ignore[16]
            container_settings = batchmodels.TaskContainerSettings(
                image_name=role.image,
                # TODO: add support for port and device bindings via container_run_options
            )
            if cfg.get("batch_container_registry"):
                # pyre-ignore[16]
                container_settings.registry = batchmodels.ContainerRegistry(
                    user_name=cfg.get("batch_container_username"),
                    password=cfg.get("batch_container_password"),
                    registry_server=cfg.get("batch_container_registry"),
                )
            task.container_settings = container_settings
        
        return task

    def _submit_dryrun(self, app: AppDef, cfg: AzureBatchOpts) -> AppDryRunInfo[AzureBatchJob]:
        name = make_unique(app.name)
        # TODO: For those cases when pool is predefined, validate that pool has right resources.
        pool_id = cfg.get("batch_pool_id") or name
        tasks = []

        for app_role in app.roles:
            task = self._role_as_task(name, app_role, cfg)
            tasks.append(task)
    
        # pyre-ignore[16]
        job = batchmodels.JobAddParameter(
            id=name,
            display_name=name,
            # pyre-ignore[16]
            pool_info=batchmodels.PoolInformation(pool_id=pool_id),
            metadata=[_as_metadata_model(k, v) for k, v in app.metadata.items()],
            on_all_tasks_complete=batchmodels.OnAllTasksComplete.terminate_job,
        )

        req = AzureBatchJob(
            job_id = name,
            pool_id = pool_id,
            job=job,
            tasks = tasks,
        )
        return AppDryRunInfo(req, repr)
    
    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        batch_service_client = self._build_client()
        job = batch_service_client.job.get(app_id)
        tasks = batch_service_client.task.list(app_id)
        roles = {}
        for task in tasks:
            env_settings  = {es.name: es.value for es in task.environment_settings}
            role_name = env_settings.get("TORCHX_ROLE_NAME", "")
            if task.multi_instance_settings:
                replicas =  task.multi_instance_settings.number_of_instances
            else:
                replicas = 0
            
            if task.container_settings:
                image = task.container_settings.image_name
            else:
                image = None
            roles[role_name] = Role(
                name=role_name,
                num_replicas=replicas,
                image=image,
                entrypoint=task.command_line,
                env=env_settings)

        return DescribeAppResponse(
            app_id=app_id,
            state=JOB_STATE[job.state], # TODO: use prev state to check
            roles=list(roles.values()),
        )

    def list(self) -> List[ListAppResponse]:
        batch_service_client = self._build_client()
        jobs = batch_service_client.job.list()
        response = []
        for job in jobs:
            response.append(ListAppResponse(app_id=job.id, state=JOB_STATE[job.state]))

        return response
   

    def _cancel_existing(self, app_id: str) -> None:
        batch_service_client = self._build_client()
        batch_service_client.job.terminate(
            job_id=app_id,
            terminate_reason="killed via torchx CLI",
        )

    def _run_opts(self) -> runopts:
        opts = runopts()
        opts.add("batch_pool_id", type_=str, help="", required=True)
        opts.add("batch_container_registry", type_=str, default="", help="", required=False)
        opts.add("batch_container_username", type_=str, default="", help="", required=False)
        opts.add("batch_container_password", type_=str, default="", help="", required=False)
    
        return opts


def create_scheduler(session_name: str,
        batch_account_name: Optional[str] = None,
        batch_account_key: Optional[str] = None,
        batch_account_url: Optional[str] = None) -> AzureBatchScheduler:
    return AzureBatchScheduler(
        session_name=session_name,
        batch_account_name=batch_account_name,
        batch_account_key=batch_account_key,
        batch_account_url=batch_account_url
    )
