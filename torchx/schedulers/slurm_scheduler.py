#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
This contains the TorchX Slurm scheduler which can be used to run TorchX
components on a Slurm cluster.
"""
import csv
import json
import logging
import os.path
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import torchx
from torchx.schedulers.api import (
    AppDryRunInfo,
    DescribeAppResponse,
    filter_regex,
    ListAppResponse,
    Scheduler,
    split_lines_iterator,
    Stream,
)
from torchx.schedulers.local_scheduler import LogIterator
from torchx.specs import (
    AppDef,
    AppState,
    macros,
    NONE,
    ReplicaStatus,
    Role,
    RoleStatus,
    runopts,
)
from torchx.workspace.dir_workspace import DirWorkspaceMixin
from typing_extensions import TypedDict

SLURM_JOB_DIRS = ".torchxslurmjobdirs"

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

SBATCH_JOB_OPTIONS = {
    "comment",
    "mail-user",
    "mail-type",
}
SBATCH_GROUP_OPTIONS = {
    "partition",
    "time",
    "constraint",
}

log: logging.Logger = logging.getLogger(__name__)


def _apply_app_id_env(s: str) -> str:
    """
    _apply_app_id_env escapes the argument and substitutes in the macros.app_id with
    a shell expression that fills in SLURM_JOB_ID from env.
    """
    escaped_parts = [shlex.quote(part) for part in s.split(macros.app_id)]
    return '"$SLURM_JOB_ID"'.join(escaped_parts)


# Using old typed dict syntax to handle hyphenated params
SlurmOpts = TypedDict(
    "SlurmOpts",
    {
        "partition": str,
        "time": str,
        "comment": Optional[str],
        "constraint": Optional[str],
        "mail-user": Optional[str],
        "mail-type": Optional[str],
        "job_dir": Optional[str],
    },
    total=False,
)


@dataclass
class SlurmReplicaRequest:
    """
    Holds parameters for a single replica running on slurm and can be materialized down to a bash script.
    """

    name: str
    entrypoint: str
    args: List[str]
    srun_opts: Dict[str, str]
    sbatch_opts: Dict[str, Optional[str]]
    env: Dict[str, str]

    @classmethod
    def from_role(
        cls, name: str, role: Role, cfg: SlurmOpts, nomem: bool
    ) -> "SlurmReplicaRequest":
        """
        ``from_role`` creates a SlurmReplicaRequest for the specific role and
        name.
        """
        sbatch_opts: Dict[str, Optional[str]] = {
            "requeue": None,
        }
        for k, v in cfg.items():
            if v is None:
                continue
            if k in SBATCH_GROUP_OPTIONS:
                sbatch_opts[k] = str(v)
        sbatch_opts.setdefault("ntasks-per-node", "1")
        resource = role.resource

        if resource != NONE:
            if resource.cpu > 0:
                sbatch_opts.setdefault("cpus-per-task", str(resource.cpu))
            if not nomem and resource.memMB > 0:
                sbatch_opts.setdefault("mem", str(resource.memMB))
            if resource.gpu > 0:
                sbatch_opts.setdefault("gpus-per-task", str(resource.gpu))

        srun_opts = {
            "output": f"slurm-{macros.app_id}-{name}.out",
            "error": f"slurm-{macros.app_id}-{name}.err",
            # kill workers N seconds after first task exits
            "wait": "60",
            # kill workers after one exits with an error
            "kill-on-bad-exit": "1",
        }

        return cls(
            name=name,
            entrypoint=role.entrypoint,
            args=list(role.args),
            sbatch_opts=sbatch_opts,
            srun_opts=srun_opts,
            env=dict(role.env),
        )

    def _opts_to_strs(self, opts: Mapping[str, Optional[str]]) -> List[str]:
        out = []
        for key, value in opts.items():
            if value is not None:
                out.append(f"--{key}={value}")
            else:
                out.append(f"--{key}")
        return out

    def materialize(self) -> Tuple[List[str], List[str]]:
        """
        materialize returns the sbatch and srun groups for this role. They
        should be combined using `:` per slurm heterogenous groups.
        """
        sbatch_args = [
            f"--job-name={self.name}",
        ] + self._opts_to_strs(self.sbatch_opts)
        srun_args = self._opts_to_strs(self.srun_opts)

        if len(self.env) > 0:
            kvs = [f"{key}={value}" for key, value in self.env.items()]
            srun_args += ["--export=ALL," + ",".join(kvs)]

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
    job_dir: Optional[str]
    max_retries: int

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
        cmd = " ".join([shlex.quote(arg) for arg in self.cmd])
        script = f"""#!/bin/bash
#
# Generated by TorchX {torchx.__version__}
# Run with: {cmd}
#
{sbatch_opts}
set -evx

export PYTHONUNBUFFERED=1
export SLURM_UNBUFFEREDIO=1
export TORCHX_MAX_RETRIES={self.max_retries}

set +e
srun {" ".join(srun_groups)}
exitcode=$?
set -e

echo "job exited with code $exitcode"
if [ $exitcode -ne 0 ]; then
    if [ "$TORCHX_MAX_RETRIES" -gt "${{SLURM_RESTART_COUNT:-0}}" ]; then
        scontrol requeue "$SLURM_JOB_ID"
    fi
    exit $exitcode
fi
"""
        return script

    def __repr__(self) -> str:
        return f"""{' '.join(self.cmd + ['$SBATCH_SCRIPT'])}

#----------------
# SBATCH_SCRIPT
#----------------
{self.materialize()}"""


class SlurmScheduler(DirWorkspaceMixin, Scheduler[SlurmOpts]):
    """
    SlurmScheduler is a TorchX scheduling interface to slurm. TorchX expects
    that slurm CLI tools are locally installed and job accounting is enabled.

    Each app def is scheduled using a heterogenous job via sbatch.
    Each replica of each role has a unique shell script generated with it's
    resource allocations and args and then sbatch is used to launch all of them
    together.

    Logs are available in combined form via ``torchx log``, the programmatic API
    as well as in the job launch directory as
    ``slurm-<jobid>-<role>-<replica_id>.out``. If TorchX is running in a
    different directory than where the job was created the logs won't be able to
    be found.

    Some of the config options passed to it are added as SBATCH arguments to each
    replica. See https://slurm.schedmd.com/sbatch.html#SECTION_OPTIONS for info
    on the arguments.

    Slurm jobs inherit the currently active ``conda`` or ``virtualenv`` and run
    in the current working directory. This matches the behavior of the
    ``local_cwd`` scheduler.

    For more info see:

    * https://slurm.schedmd.com/sbatch.html
    * https://slurm.schedmd.com/heterogeneous_jobs.html

    .. code-block:: bash

        $ torchx run --scheduler slurm utils.echo --msg hello
        slurm://torchx_user/1234
        $ torchx status slurm://torchx_user/1234
        $ less slurm-1234.out
        ...

    **Config Options**

    .. runopts::
        class: torchx.schedulers.slurm_scheduler.create_scheduler

    **Compatibility**

    .. compatibility::
        type: scheduler
        features:
            cancel: true
            logs: true
            distributed: true
            describe: |
                Partial support. SlurmScheduler will return job and replica
                status but does not provide the complete original AppSpec.
            workspaces: |
                If ``job_dir`` is specified the DirWorkspaceMixin will create a new
                isolated directory with a snapshot of the workspace.
            mounts: false
            elasticity: false

    If a partition has less than 1GB of RealMemory configured we disable memory
    requests to workaround https://github.com/aws/aws-parallelcluster/issues/2198.
    """

    def __init__(self, session_name: str) -> None:
        # NOTE: make sure any new init options are supported in create_scheduler(...)
        super().__init__("slurm", session_name)

    def _run_opts(self) -> runopts:
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
            help='The maximum time the job is allowed to run for. Formats: \
            "minutes", "minutes:seconds", "hours:minutes:seconds", "days-hours", \
            "days-hours:minutes" or "days-hours:minutes:seconds"',
        )
        opts.add(
            "comment",
            type_=str,
            help="Comment to set on the slurm job.",
        )
        opts.add(
            "constraint",
            type_=str,
            help="Constraint to use for the slurm job.",
        )
        opts.add(
            "mail-user",
            type_=str,
            help="User to mail on job end.",
        )
        opts.add(
            "mail-type",
            type_=str,
            help="What events to mail users on.",
        )
        opts.add(
            "job_dir",
            type_=str,
            help="""The directory to place the job code and outputs. The
            directory must not exist and will be created. To enable log
            iteration, jobs will be tracked in ``.torchxslurmjobdirs``.
            """,
        )
        return opts

    def schedule(self, dryrun_info: AppDryRunInfo[SlurmBatchRequest]) -> str:
        req = dryrun_info.request
        job_dir = req.job_dir
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(job_dir or tmpdir, "torchx-sbatch.sh")
            if job_dir is not None:
                req.cmd += [f"--chdir={job_dir}"]
            req.cmd += [path]
            script = req.materialize()

            with open(path, "w") as f:
                f.write(script)

            p = subprocess.run(req.cmd, stdout=subprocess.PIPE, check=True)
            job_id = p.stdout.decode("utf-8").strip()

            if job_dir is not None:
                _save_job_dir(job_id, job_dir)

            return job_id

    def _partition_memmb(self, partition: Optional[str]) -> Optional[int]:
        """
        _partition_memmb returns the memory allocation for the given partition
        or the default partition if none is specified.
        """
        try:
            p = subprocess.run(
                ["sinfo", "--format", "%P,%m", "--noconvert"],
                stdout=subprocess.PIPE,
            )
        except FileNotFoundError:
            return None
        if p.returncode != 0:
            return None
        output = p.stdout.decode("utf-8").strip().split("\n")
        if len(output) <= 1:
            return None

        reader = csv.DictReader(output, delimiter=",")
        for row in reader:
            part = row.get("PARTITION")
            mem = row.get("MEMORY")
            if part is None or mem is None:
                continue
            default = "*" in part
            part = part.strip("*")
            memmb = int(mem.strip("M+"))
            if part == partition or (partition is None and default):
                return memmb
        return None

    def _submit_dryrun(
        self, app: AppDef, cfg: SlurmOpts
    ) -> AppDryRunInfo[SlurmBatchRequest]:
        job_dir = cfg.get("job_dir")
        assert job_dir is None or isinstance(job_dir, str), "job_dir must be str"

        partition = cfg.get("partition")
        assert partition is None or isinstance(partition, str), "partition must be str"

        # check if the partition has at least 1GB memory, if we're not sure,
        # default to using memory allocations
        memmb = self._partition_memmb(partition)
        nomem = memmb is not None and memmb <= 1000

        replicas = {}
        for role in app.roles:
            for replica_id in range(role.num_replicas):
                values = macros.Values(
                    img_root=role.image,
                    app_id=macros.app_id,
                    replica_id=str(replica_id),
                    rank0_env="SLURM_JOB_NODELIST_HET_GROUP_0",
                )
                name = f"{role.name}-{replica_id}"
                replica_role = values.apply(role)
                replicas[name] = SlurmReplicaRequest.from_role(
                    name,
                    replica_role,
                    cfg,
                    nomem=nomem,
                )
        cmd = ["sbatch", "--parsable"]

        for k in SBATCH_JOB_OPTIONS:
            # pyre-fixme: Typed Dict requires string literal
            if k in cfg and cfg[k] is not None:
                # pyre-fixme: Typed Dict requires string literal
                cmd += [f"--{k}={cfg[k]}"]

        req = SlurmBatchRequest(
            cmd=cmd,
            replicas=replicas,
            job_dir=job_dir,
            max_retries=min(role.max_retries for role in app.roles),
        )

        return AppDryRunInfo(req, repr)

    def _validate(self, app: AppDef, scheduler: str) -> None:
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

            role, _, replica_id = row["JobName"].rpartition("-")
            if not replica_id or not role:
                # name should always have at least 3 parts but sometimes sacct
                # is slow to update
                continue
            if role not in roles:
                roles[role] = Role(name=role, num_replicas=0, image="")
                roles_statuses[role] = RoleStatus(role, [])
            roles[role].num_replicas += 1
            roles_statuses[role].replicas.append(
                ReplicaStatus(
                    id=int(replica_id), role=role, state=app_state, hostname=""
                ),
            )

        return DescribeAppResponse(
            app_id=app_id,
            roles=list(roles.values()),
            roles_statuses=list(roles_statuses.values()),
            state=app_state,
            msg=msg,
        )

    def log_iter(
        self,
        app_id: str,
        role_name: str,
        k: int = 0,
        regex: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        should_tail: bool = False,
        streams: Optional[Stream] = None,
    ) -> Iterable[str]:
        if streams is None:
            log.info("log stream not specified, defaulting to STDERR")
        elif streams == Stream.COMBINED:
            raise ValueError(
                "SlurmScheduler does not support COMBINED log stream."
                " Use `stdout` or `stderr`"
            )

        extension = "out" if streams == Stream.STDOUT else "err"

        if since or until:
            log.warning(
                "since and/or until times specified for SlurmScheduler.log_iter."
                " These will be ignored and all log lines will be returned"
            )

        log_file = f"slurm-{app_id}-{role_name}-{k}.{extension}"
        job_dirs = _get_job_dirs()
        if app_id in job_dirs:
            log_file = os.path.join(job_dirs[app_id], log_file)

        iterator = LogIterator(app_id, log_file, self, should_tail=should_tail)
        # sometimes there's multiple lines per logged line
        iterator = split_lines_iterator(iterator)
        if regex:
            iterator = filter_regex(regex, iterator)
        return iterator

    def list(self) -> List[ListAppResponse]:
        # By default sacct only returns accounting information of jobs launched on the current day
        # To return all jobs launched, set starttime to one second past unix epoch time
        # Starttime will be modified when listing jobs by timeframe is supported
        p = subprocess.run(
            ["sacct", "--json", "-S1970-01-01-00:00:01"],
            stdout=subprocess.PIPE,
            check=True,
        )
        output_json = json.loads(p.stdout.decode("utf-8"))
        return [
            ListAppResponse(
                app_id=str(job["job_id"]), state=SLURM_STATES[job["state"]["current"]]
            )
            for job in output_json["jobs"]
        ]


def create_scheduler(session_name: str, **kwargs: Any) -> SlurmScheduler:
    return SlurmScheduler(
        session_name=session_name,
    )


def _save_job_dir(job_id: str, job_dir: str) -> None:
    with open(SLURM_JOB_DIRS, "at") as f:
        f.write(f"{job_id} = {job_dir}\n")


def _get_job_dirs() -> Mapping[str, str]:
    try:
        with open(SLURM_JOB_DIRS, "rt") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return {}

    out = {}
    for line in lines:
        first, _, second = line.partition("=")
        if not first or not second:
            continue
        out[first.strip()] = second.strip()
    return out
