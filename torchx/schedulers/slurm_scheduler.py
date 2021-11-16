#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This contains the TorchX Slurm scheduler which can be used to run TorchX
components on a Slurm cluster.
"""

import csv
import os.path
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from torchx.schedulers.api import AppDryRunInfo, DescribeAppResponse, Scheduler
from torchx.specs import (
    NONE,
    AppDef,
    AppState,
    CfgVal,
    ReplicaStatus,
    Role,
    RoleStatus,
    SchedulerBackend,
    macros,
    runopts,
)


SLURM_STATES: Mapping[str, AppState] = {
    "BOOT_FAIL": AppState.FAILED,
    "CANCELLED": AppState.CANCELLED,
    "COMPLETED": AppState.SUCCEEDED,
    "DEADLINE": AppState.FAILED,
    "FAILED": AppState.FAILED,
    "NODE_FAIL": AppState.FAILED,
    "OUT_OF_MEMORY": AppState.FAILED,
    "PENDING": AppState.PENDING,
    "PREEMPTED": AppState.FAILED,
    "RUNNING": AppState.RUNNING,
    "REQUEUED": AppState.PENDING,
    "RESIZING": AppState.PENDING,
    "REVOKED": AppState.FAILED,
    "SUSPENDED": AppState.PENDING,
    "TIMEOUT": AppState.FAILED,
}


def _apply_app_id_env(s: str) -> str:
    """
    _apply_app_id_env escapes the argument and substitutes in the macros.app_id with
    a shell expression that fills in SLURM_JOB_ID from env.
    """
    escaped_parts = [shlex.quote(part) for part in s.split(macros.app_id)]
    return '"$SLURM_JOB_ID"'.join(escaped_parts)


@dataclass
class SlurmReplicaRequest:
    """
    Holds parameters for a single replica running on slurm and can be materialized down to a bash script.
    """

    name: str
    dir: str
    entrypoint: str
    args: List[str]
    srun_opts: Dict[str, str]
    sbatch_opts: Dict[str, str]
    env: Dict[str, str]

    @classmethod
    def from_role(
        cls, name: str, role: Role, cfg: Mapping[str, CfgVal]
    ) -> "SlurmReplicaRequest":
        sbatch_opts = {k: str(v) for k, v in cfg.items() if v is not None}
        sbatch_opts.setdefault("ntasks-per-node", "1")
        resource = role.resource

        if resource != NONE:
            if resource.cpu > 0:
                sbatch_opts.setdefault("cpus-per-task", str(resource.cpu))
            if resource.memMB > 0:
                sbatch_opts.setdefault("mem", str(resource.memMB))
            if resource.gpu > 0:
                sbatch_opts.setdefault("gpus-per-task", str(resource.gpu))

        return cls(
            name=name,
            dir=role.image,
            entrypoint=role.entrypoint,
            args=list(role.args),
            sbatch_opts=sbatch_opts,
            srun_opts={},
            env=dict(role.env),
        )

    def materialize(self) -> Tuple[List[str], List[str]]:
        """
        materialize returns the sbatch and srun groups for this role. They
        should be combined using `:` per slurm heterogenous groups.
        """
        sbatch_args = [
            f"--job-name={self.name}",
        ] + [f"--{key}={value}" for key, value in self.sbatch_opts.items()]
        srun_args = (
            [f"--chdir={self.dir}"]
            + [f"--{key}={value}" for key, value in self.srun_opts.items()]
            + [f"--export={key}={value}" for key, value in self.env.items()]
        )

        srun_group = srun_args + [self.entrypoint] + self.args
        srun_group = [_apply_app_id_env(arg) for arg in srun_group]

        return sbatch_args, srun_group


@dataclass
class SlurmBatchRequest:
    """
    Holds parameters used to launch a slurm job via sbatch.
    """

    cmd: List[str]
    replicas: Dict[str, SlurmReplicaRequest]

    def materialize(self) -> str:
        """
        materialize returns the contents of the script that can be passed to
        sbatch to run the job.
        """

        sbatch_groups = []
        srun_groups = []
        for i, replica in enumerate(self.replicas.values()):
            if i > 0:
                srun_groups.append(":\\\n    ")

            sbatch_group, srun_group = replica.materialize()
            sbatch_groups.append(" ".join(sbatch_group))
            srun_groups += srun_group

        sbatch_opts = "#SBATCH hetjob\n".join(
            f"#SBATCH {group}\n" for group in sbatch_groups
        )
        script = f"""#!/bin/bash
{sbatch_opts}
# exit on error
set -e

srun {" ".join(srun_groups)}
"""
        sbatch_cmd = self.cmd + sbatch_groups
        return script


class SlurmScheduler(Scheduler):
    """
    SlurmScheduler is a TorchX scheduling interface to slurm. TorchX expects
    that slurm CLI tools are locally installed and job accounting is enabled.

    Each app def is scheduled using a heterogenous job via sbatch.
    Each replica of each role has a unique shell script generated with it's
    resource allocations and args and then sbatch is used to launch all of them
    together.

    Logs are written to the default slurm log file.

    Any scheduler options passed to it are added as SBATCH arguments to each
    replica. See https://slurm.schedmd.com/sbatch.html#SECTION_OPTIONS for info
    on the arguments.

    For more info see:

    * https://slurm.schedmd.com/sbatch.html
    * https://slurm.schedmd.com/heterogeneous_jobs.html

    .. code-block:: bash

        $ torchx run --scheduler slurm utils.echo --msg hello
        slurm://torchx_user/1234
        $ torchx status slurm://torchx_user/1234
        $ less slurm-1234.out
        ...

    .. compatibility::
        type: scheduler
        features:
            cancel: true
            logs: |
                Logs are accessible via the default slurm log file but not the
                programmatic API.
            distributed: true
            describe: |
                Partial support. SlurmScheduler will return job and replica
                status but does not provide the complete original AppSpec.
    """

    def __init__(self, session_name: str) -> None:
        super().__init__("slurm", session_name)

    def run_opts(self) -> runopts:
        opts = runopts()
        opts.add(
            "partition",
            type_=str,
            help="The partition to run the job in.",
            default=None,
        )
        opts.add(
            "time",
            type_=str,
            default=None,
            help="The maximum time the job is allowed to run for.",
        )
        return opts

    def schedule(self, dryrun_info: AppDryRunInfo[SlurmBatchRequest]) -> str:
        req = dryrun_info.request
        with tempfile.TemporaryDirectory() as tmpdir:
            script = req.materialize()
            path = os.path.join(tmpdir, "job.sh")

            with open(path, "w") as f:
                f.write(script)

            cmd = req.cmd + [path]

            p = subprocess.run(cmd, stdout=subprocess.PIPE, check=True)
            return p.stdout.decode("utf-8").strip()

    def _submit_dryrun(
        self, app: AppDef, cfg: Mapping[str, CfgVal]
    ) -> AppDryRunInfo[SlurmBatchRequest]:
        replicas = {}
        for role in app.roles:
            for replica_id in range(role.num_replicas):
                values = macros.Values(
                    img_root=role.image,
                    app_id=macros.app_id,
                    replica_id=str(replica_id),
                )
                name = f"{app.name}-{role.name}-{replica_id}"
                replica_role = values.apply(role)
                replicas[name] = SlurmReplicaRequest.from_role(name, replica_role, cfg)
        req = SlurmBatchRequest(
            cmd=["sbatch", "--parsable"],
            replicas=replicas,
        )
        return AppDryRunInfo(req, repr)

    def _validate(self, app: AppDef, scheduler: SchedulerBackend) -> None:
        # Skip validation step for slurm
        pass

    def _cancel_existing(self, app_id: str) -> None:
        subprocess.run(["scancel", app_id], check=True)

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        p = subprocess.run(
            ["sacct", "--parsable2", "-j", app_id], stdout=subprocess.PIPE, check=True
        )
        output = p.stdout.decode("utf-8").split("\n")
        if len(output) <= 1:
            return None

        reader = csv.DictReader(output, delimiter="|")

        roles = {}
        roles_statuses = {}
        msg = ""
        app_state = AppState.UNKNOWN
        for row in reader:
            job_id, *parts = row["JobID"].split("+")
            if job_id != app_id:
                continue
            if len(parts) > 0 and "." in parts[0]:
                # we only care about the worker not the child jobs
                continue

            state = row["State"]
            msg = state
            state_enum = SLURM_STATES.get(state)
            assert (
                state_enum
            ), f"failed to translate slurm state {state} to torchx state"
            app_state = state_enum

            name_parts = row["JobName"].split("-")
            if len(name_parts) < 3:
                # name should always have at least 3 parts but sometimes sacct
                # is slow to update
                continue
            role = name_parts[-2]
            replica_id = int(name_parts[-1])
            if role not in roles:
                roles[role] = Role(name=role, num_replicas=0, image="")
                roles_statuses[role] = RoleStatus(role, [])
            roles[role].num_replicas += 1
            roles_statuses[role].replicas.append(
                ReplicaStatus(id=replica_id, role=role, state=app_state, hostname=""),
            )

        return DescribeAppResponse(
            app_id=app_id,
            roles=list(roles.values()),
            roles_statuses=list(roles_statuses.values()),
            state=app_state,
            msg=msg,
        )


def create_scheduler(session_name: str, **kwargs: Any) -> SlurmScheduler:
    return SlurmScheduler(
        session_name=session_name,
    )
