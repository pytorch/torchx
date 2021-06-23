#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import csv
import os.path
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional

from torchx.schedulers.api import AppDryRunInfo, DescribeAppResponse, Scheduler
from torchx.specs.api import (
    NONE,
    AppDef,
    AppState,
    Role,
    RunConfig,
    SchedulerBackend,
    macros,
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


def _slurm_escape(s: str) -> str:
    """
    _slurm_escape escapes the argument and substitutes in the macros.app_id with
    a shell expression that fills in SLURM_JOB_ID from env.
    """
    escaped_parts = [shlex.quote(part) for part in s.split(macros.app_id)]
    return '"$SLURM_JOB_ID"'.join(escaped_parts)


@dataclass
class SlurmReplicaRequest:
    """
    Holds parameters for a single replica running on slurm and can be materialized down to a bash script.
    """

    dir: str
    entrypoint: str
    args: List[str]
    opts: Dict[str, str]
    env: Dict[str, str]

    @classmethod
    def from_role(cls, role: Role, cfg: RunConfig) -> "SlurmReplicaRequest":
        opts = {k: str(v) for k, v in cfg.cfgs.items()}

        if (resource := role.resource) != NONE:
            if (cpu := resource.cpu) > 0:
                opts["cpus-per-task"] = str(cpu)
            if (memMB := resource.memMB) > 0:
                opts["mem"] = str(memMB)
            if (gpu := resource.gpu) > 0:
                opts["gpus-per-task"] = str(gpu)

        return cls(
            dir=role.image,
            entrypoint=role.entrypoint,
            args=list(role.args),
            opts=opts,
            env=dict(role.env),
        )

    def materialize(self) -> str:
        sbatch_opts = [f"#SBATCH --{key}={value}" for key, value in self.opts.items()]
        sbatch_opts += [
            f"#SBATCH --export={key}={value}" for key, value in self.env.items()
        ]
        sbatch_opts_str = "\n".join(sbatch_opts)

        escaped_args = [_slurm_escape(arg) for arg in self.args]

        return f"""#!/bin/sh
{sbatch_opts_str}

# exit on error
set -e

srun --chdir={self.dir} {self.entrypoint} {" ".join(escaped_args)}
"""


@dataclass
class SlurmBatchRequest:
    """
    Holds parameters used to launch a slurm job via sbatch.
    """

    cmd: List[str]
    replicas: Dict[str, SlurmReplicaRequest]


class SlurmScheduler(Scheduler):
    """
    SlurmScheduler is a TorchX scheduling interface to slurm. TorchX expects
    that slurm CLI tools are locally installed and job accounting is enabled.

    Each app def is scheduled using a heterogenous job via sbatch.
    Each replica of each role has a unique shell script generated with it's
    resource allocations and args and then sbatch is used to launch all of them
    together.

    Logs are written to the default slurm log file.

    Any scheduler options passed to it are added as SBATCH arguments to each replica.

    For more info see:

    * https://slurm.schedmd.com/sbatch.html
    * https://slurm.schedmd.com/heterogeneous_jobs.html

    .. code-block:: bash

        $ torchx run --scheduler slurm utils.echo --msg hello
        slurm://torchx_user/1234
        $ torchx status slurm://torchx_user/1234
        $ less slurm-1234.out
        ...
    """

    def __init__(self, session_name: str) -> None:
        super().__init__("slurm", session_name)

    def schedule(self, dryrun_info: AppDryRunInfo[SlurmBatchRequest]) -> str:
        req = dryrun_info.request
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, (name, body) in enumerate(req.replicas.items()):
                path = os.path.join(tmpdir, name)
                with open(path, "w") as f:
                    f.write(body.materialize())

                if i > 0:
                    req.cmd.append(":")
                req.cmd.append(path)

            p = subprocess.run(req.cmd, stdout=subprocess.PIPE, check=True)
            return p.stdout.decode("utf-8").strip()

    def _submit_dryrun(
        self, app: AppDef, cfg: RunConfig
    ) -> AppDryRunInfo[SlurmBatchRequest]:
        cmd = ["sbatch", "--parsable", "--job-name", app.name]
        replicas = {}
        for i, role in enumerate(app.roles):
            for replica_id in range(role.num_replicas):
                values = macros.Values(
                    img_root=role.image,
                    app_id=macros.app_id,
                    replica_id=str(replica_id),
                )
                name = f"role-{i}-{role.name}-{replica_id}.sh"
                replica_role = values.apply(role)
                replicas[name] = SlurmReplicaRequest.from_role(replica_role, cfg)
        req = SlurmBatchRequest(
            cmd=cmd,
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

        resp = DescribeAppResponse(
            app_id=app_id,
        )
        for row in reader:
            if row["JobID"] == app_id:
                state = row["State"]
                resp.msg = state
                state_enum = SLURM_STATES.get(state)
                assert (
                    state_enum
                ), f"failed to translate slurm state {state} to torchx state"
                resp.state = state_enum

        return resp


def create_scheduler(session_name: str, **kwargs: Any) -> SlurmScheduler:
    return SlurmScheduler(
        session_name=session_name,
    )
