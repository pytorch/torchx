#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This contains the TorchX LSF scheduler which can be used to run TorchX
components on a LSF cluster.
"""
import csv
import json
import logging
import os.path
import shlex
import subprocess
import tempfile
import re
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
from torchx.schedulers.ids import make_unique
from torchx.schedulers.local_scheduler import LogIterator
from torchx.specs import (
    AppDef,
    AppState,
    BindMount,
    DeviceMount,
    VolumeMount,
    macros,
    NONE,
    ReplicaStatus,
    Role,
    RoleStatus,
    runopts,
)
from torchx.workspace.dir_workspace import DirWorkspace
from typing_extensions import TypedDict

JOB_STATE: Dict[str, AppState] = {
    "DONE": AppState.SUCCEEDED,
    "EXIT": AppState.FAILED,
    "PEND": AppState.PENDING,
    "RUN": AppState.RUNNING,
    "PSUSP": AppState.PENDING,
    "USUSP": AppState.PENDING,
    "SSUSP": AppState.PENDING,
}

def get_job_state(state_str: str, exit_code: str) -> AppState:
    state = AppState.UNKNOWN
    if state_str in JOB_STATE.keys():
        state = JOB_STATE[state_str]
    if state == AppState.FAILED and exit_code == "130": # likely SIGINT
        state = AppState.CANCELLED
    return state

class LsfOpts(TypedDict, total=False):
    lsf_queue: Optional[str]
    jobdir: Optional[str]   # NOTE: *job_dir* cannot be used. somehow it overwrites --image flag (bug?). so use *jobdir* (without underscore, _)
    runtime: str
    container_workdir: Optional[str]
    host_network: Optional[bool]
    shm_size: Optional[str]

def get_docker_command(job_name: str, role: Role, cfg: LsfOpts) -> str:
    cmds = ["docker", "run", f"--name={job_name}"]
    for mount in role.mounts:
        rw = "rw"
        if mount.read_only:
            rw = "ro"
        if isinstance(mount, BindMount):
            cmds += ["-v", f"{mount.src_path}:{mount.dst_path}:{rw}"]
        elif isinstance(mount, VolumeMount):
            cmds += ["--mount", f"\"type=volume,src={mount.src_path}:dst={mount.dst_path},{rw}\""]
        elif isinstance(mount, DeviceMount):
            cmds += [f"--device={mount.src_path}:{mount.dst_path}:{mount.permissions}"]
        else:
            raise Exception(f"Unknown mount type: {mount}")
    container_workdir = cfg.get("container_workdir")
    if container_workdir:
        cmds += ["-w", container_workdir]
    host_network = cfg.get("host_network")
    if host_network:
        cmds += ["--net=host", "--ipc=host"]
    else:
        for name, port in role.port_map.items():
            cmds += ["-p", str(port)]
    shm_size = cfg.get("shm_size")
    if shm_size:
        cmds += [f"--shm-size={shm_size}"]
    for key, value in dict(role.env).items():
        cmds += ["-e", f"{key}=\"{value}\""]

    resource = role.resource
    if resource != NONE:
        if resource.cpu > 0:
            cmds += [f"--cpus={str(resource.cpu)}"]
        if resource.memMB > 0:
            cmds += [f"--memory={str(resource.memMB * 1024 * 1024)}"]
        if resource.gpu > 0:
            cmds += [f"--gpus", "all"]
    cmds += ["--entrypoint", role.entrypoint, "--rm", role.image] + [shlex.quote(arg) for arg in role.args]
    return " ".join(cmds).replace("$", "\\$")

def get_singularity_command(job_name: str, role: Role, cfg: LsfOpts) -> str:
    cmds = []
    for key, value in dict(role.env).items():
        cmds += [f"{key}=\"{value}\""]
    cmds += ["singularity", "exec", "--nv"]
    for mount in role.mounts:
        rw = "rw"
        if mount.read_only:
            rw = "ro"
        if isinstance(mount, BindMount):
            cmds += ["--bind", f"{mount.src_path}:{mount.dst_path}:{rw}"]
        else:
            raise Exception(f"Unknown/unsupported mount type: {mount}")
    container_workdir = cfg.get("container_workdir")
    if container_workdir:
        cmds += ["-W", container_workdir]

    cmds += [role.image, role.entrypoint] + [shlex.quote(arg) for arg in role.args]
    return " ".join(cmds).replace("$", "\\$")


def get_native_command(job_name: str, role: Role, cfg: LsfOpts) -> str:
    cmds = []
    for key, value in dict(role.env).items():
        cmds += [f"{key}=\"{value}\""]
    cmds += [role.entrypoint] + [shlex.quote(arg) for arg in role.args]
    return " ".join(cmds).replace("$", "\\$")

def get_command(job_name: str, role: Role, cfg: LsfOpts) -> str:
    runtime = cfg.get("runtime")
    if runtime == "native":
        return get_native_command(job_name, role, cfg)
    if runtime == "docker":
        return get_docker_command(job_name, role, cfg)
    if runtime == "singularity":
        return get_singularity_command(job_name, role, cfg)
    raise Exception("Unkonwn runtime: " + runtime)

def get_bsub(app_id:str, job_name: str, role: Role, cfg: LsfOpts, head_job_name: str, head_job_host: str) -> str:
    bsub_args = ["bsub", "-P", app_id, "-J", job_name]
    if head_job_name != "":
        bsub_args += ["-w", f"\"started({head_job_name})\"", "-R", f"\"select[hname!='{head_job_host}']\""]
    else:
        bsub_args += ["-m", head_job_host]
    jobdir = cfg.get("jobdir")
    if jobdir:
        bsub_args += ["-cwd", jobdir, "-outdir", jobdir, "-oo", f"{jobdir}/{job_name}.submit.out", "-eo", f"{jobdir}/{job_name}.submit.err"]
    for k, v in cfg.items():
        if v is None:
            continue
        if k == "lsf_queue":
            bsub_args += ["-q", v]
    resource = role.resource

    if resource != NONE:
        if resource.cpu > 0:
            bsub_args += ["-n", str(resource.cpu)]
        if resource.memMB > 0:
            bsub_args += ["-R", f"\"span[hosts=1] rusage[mem={str(resource.memMB)}]\""]
        else:
            bsub_args += ["-R", f"\"span[hosts=1]]\""]
        if resource.gpu > 0:
            bsub_args += ["-gpu", f"\"num={str(resource.gpu)}:mode=shared:j_exclusive=yes\""]
    bsub_line = " ".join(bsub_args)
    if jobdir:
        return f"{bsub_line} << EOF\n{get_command(job_name, role, cfg)} > {jobdir}/{job_name}.out 2> {jobdir}/{job_name}.err\nEOF"
    else:
        return f"{bsub_line} << EOF\n{get_command(job_name, role, cfg)}\nEOF"

def cleanup_str(data: str) -> str:
    """
    Invokes ``lower`` on thes string and removes all
    characters that do not satisfy ``[a-z0-9]`` pattern.
    This method is mostly used to make sure kubernetes scheduler gets
    the job name that does not violate its validation.
    """
    if data.startswith("-"):
        data = data[1:]
    pattern = r"[a-z0-9\-_]"
    return "".join(re.findall(pattern, data.lower()))

def find_rank0_host(role: Role) -> str:
    resource = role.resource
    cpu = 1
    gpu = 0

    if resource != NONE:
        if resource.cpu > 0:
            cpu = resource.cpu
        if resource.gpu > 0:
            gpu = resource.gpu

    p = subprocess.run(["bhosts", "-noheader", "-o", "hname max ng"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
    msg = p.stdout.decode("utf-8")
    for line in msg.split("\n"):
        split = line.split(" ")
        try:
            if len(split) >= 3 and cpu <= int(split[1]) and gpu <= int(split[2]):
                return split[0]
        except:
            continue
    raise Exception(f"cannot find a host with {cpu} CPUs, and {gpu} GPUs. Try again with enough available resource.")

def get_submit_script(app_id: str, cmd: List[str], app: AppDef, cfg: LsfOpts) -> Tuple[str, str]:
    bsubs = []
    rank0_host = ""
    head_job_name = ""
    for role_idx, role in enumerate(app.roles):
        if role_idx == 0:
            rank0_host = find_rank0_host(role)
        for replica_id in range(role.num_replicas):
            values = macros.Values(
                img_root="",
                app_id=app_id,
                replica_id=str(replica_id),
                rank0_env=f"TORCHX_RANK0_HOST",
            )
            name = cleanup_str(f"{role.name}-{replica_id}")
            replica_role = values.apply(role)
            replica_role.env["TORCHX_RANK0_HOST"] = rank0_host
            job_name = app_id + "-" + name
            bsubs.append(get_bsub(app_id, job_name, replica_role, cfg, head_job_name, rank0_host))
            if role_idx == 0 and replica_id == 0:
                head_job_name = job_name
    cmd = " ".join([shlex.quote(arg) for arg in cmd])
    script = f"""#!/bin/bash
#
# Generated by TorchX {torchx.__version__}
# Run with: {cmd}
#
"""
    return script + "\n".join(bsubs) + "\n"

@dataclass
class LsfBsub:
    jobdir: str
    app_id: str
    app: AppDef
    cfg: LsfOpts

    cmd: List[str]

    def materialize(self) -> str:
        return get_submit_script(self.app_id, self.cmd, self.app, self.cfg)

    def __repr__(self) -> str:
        return f"""{' '.join(self.cmd + ['$BSUB_SCRIPT'])}
#------------
# BSUB_SCRIPT
#------------
{self.materialize()}"""

class LsfScheduler(Scheduler[LsfOpts], DirWorkspace):
    """
    **Example: native hello_world and common utils**
    .. code-block:: bash
        $ torchx run -s lsf -cfg jobdir=/mnt/data/torchx,runtime=native utils.echo --msg hello_world --num_replicas 3
        lsf://torchx/echo-pxc3gn5ct061k
        $ torchx list -s lsf
        $ torchx status lsf://torchx/echo-pxc3gn5ct061k
        $ torchx cancel lsf://torchx/echo-pxc3gn5ct061k
        $ torchx log --stream stdout lsf://torchx/echo-pxc3gn5ct061k/echo/0  #Note: jobdir needs to be shared among all nodes (e.g., NFS) to use this functionality
        ...

    **Example: Docker hello_world**
    .. code-block:: bash
        $ torchx run -s lsf -cfg jobdir=/mnt/data/torchx,runtime=docker utils.echo --image alpine:latest --msg hello_world --num_replicas 3
        ...

    **Example: Singularity hello_world**
    .. code-block:: bash
        $ torchx run -s lsf -cfg jobdir=/mnt/data/torchx,runtime=singularity utils.echo --image docker://alpine:latest --msg hello_world --num_replicas 3
        ...

    **Example: Docker Gloo**
    .. code-block:: bash
        $ cp dist_app.py /mnt/data/dist/
        $ torchx run -s lsf -cfg "jobdir=/mnt/data/torchx,runtime=docker,host_network=True" dist.ddp -j 2x2 --gpu 2 --script /data/dist_app.py --mount "type=bind,src=/mnt/data/dist,dst=/data"
        ...

    **Example: Singularity Gloo**
    .. code-block:: bash
        $ cp dist_app.py /mnt/data/dist/
        $ torchx run -s lsf -cfg "jobdir=/mnt/data/torchx,runtime=singularity,host_network=True" dist.ddp --image docker://ghcr.io/pytorch/torchx:0.3.0dev0 -j 2x2 --gpu 2 --script /data/dist_app.py --mount "type=bind,src=/mnt/data/dist,dst=/data"
        ...
    **Config Options**
    .. runopts::
        class: torchx.schedulers.lsf_scheduler.create_scheduler

    **Compatibility**

    .. compatibility::
        type: scheduler
        features:
          cancel: true
          logs: true
          distributed: true
          describe: |
              LsfScheduler will return job.
          mounts: true

    **TOFIX**

    - On host_network=False, tasks cannot reoslve static names such as /etc/hosts (containers cannot reach it without host network)
    - Image downloads should be separated jobs

    """

    def __init__(self, session_name: str) -> None:
        super().__init__("lsf", session_name)

    def run_opts(self) -> runopts:
        opts = runopts()
        opts.add(
            "lsf_queue",
            type_=str,
            default=None,
            help='queue name to submit jobs',
        )
        opts.add(
            "jobdir",
            type_=str,
            default=None,
            help='The directory to place the job code and outputs. The directory must not exist and will be created.',
        )
        opts.add(
            "runtime",
            type_=str,
            default=None,
            help='container name (docker|native)',
        )
        opts.add(
            "container_workdir",
            type_=str,
            default=None,
            help='working directory in container jobs',
        )
        opts.add(
            "host_network",
            type_=bool,
            default=False,
            help='True if using the host network for jobs',
        )
        opts.add(
            "shm_size",
            type_=str,
            default="64m",
            help='size of shared memory (/dev/shm) for jobs',
        )

        return opts

    def schedule(self, dryrun_info: AppDryRunInfo[LsfBsub]) -> str:
        req = dryrun_info.request
        jobdir = req.jobdir
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(jobdir or tempdir, f"{req.app_id}.sh")
            req.cmd += [path]
            script = req.materialize()
            with open(path, "w") as f:
                f.write(script)
            p = subprocess.run(req.cmd, stdout=subprocess.PIPE, check=True)
            job_ret = p.stdout.decode("utf-8").strip()
        return req.app_id


    def _validate(self, app: AppDef, scheduler: str) -> None:
        # Skip validation step for lsf
        pass

    def _cancel_existing(self, app_id: str) -> None:
        p = subprocess.run(["bjobs", "-noheader", "-a", "-P", app_id, "-o", "id"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        msg = p.stdout.decode("utf-8")
        if msg == "":
            return None
        subprocess.run(["bkill"] + msg.strip().split("\n"), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)

    def _submit_dryrun(self, app: AppDef, cfg: LsfBsub) -> AppDryRunInfo[LsfBsub]:
        jobdir = cfg.get("jobdir")
        assert jobdir is None or isinstance(jobdir, str), "jobdir must be str"

        app_id = cleanup_str(make_unique(app.name))
        return AppDryRunInfo(LsfBsub(app_id=app_id, cmd=["/bin/bash"], jobdir=jobdir, app=app, cfg=cfg), repr)

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        p = subprocess.run(["bjobs", "-noheader", "-a", "-P", app_id, "-o", "proj name stat exit_code"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        msg = p.stdout.decode("utf-8")
        if msg == "":
            return None
        roles = {}
        roles_statuses = {}
        app_state = AppState.RUNNING
        success_count = 0
        total_count = 0
        for line in msg.split("\n"):
            split = line.split(" ")
            if len(split) < 4:
                continue
            proj = split[0]
            role, _, idx = split[1][len(proj)+1:].rpartition("-")
            idx = int(idx)
            if role not in roles:
                roles[role] = Role(name=role, num_replicas=0, image="")
                roles_statuses[role] = RoleStatus(role, [])
            roles[role].num_replicas += 1
            state = get_job_state(split[2], split[3])
            roles_statuses[role].replicas.append(ReplicaStatus(id=idx, role=role, state=state, hostname=""))
            if state == AppState.FAILED or state == AppState.CANCELLED or state == AppState.PENDING or state == AppState.UNKNOWN:
                app_state = state
            elif state == AppState.SUCCEEDED:
                success_count += 1
            total_count += 1
        if success_count == total_count:
            app_state = AppState.SUCCEEDED
        # set roles, roles_statuses, app_state
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
        if streams == Stream.COMBINED:
            raise ValueError(
                "LsfScheduler does not support COMBINED log stream."
                " Use `stdout` or `stderr`"
            )

        extension = "err" if streams == Stream.STDERR else "out"

        p = subprocess.run(["bjobs", "-noheader", "-a", "-P", app_id, "-o", "proj name output_dir"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        lines = p.stdout.decode("utf-8").split("\n")
        jobs = {}
        log_file = ""
        for line in lines:
            if line != "":
                split = line.split(" ")
                if split[2] == "-":
                    continue
                proj = split[0]
                role, _, idx = split[1][len(proj)+1:].rpartition("-")
                if app_id == proj and role == role_name and idx == str(k):
                    log_file = split[2] + f"/{split[1]}.{extension}"
        if log_file == "":
            raise ValueError(f"cannot find log directory for {app_id}. Note: need to specify -cfg jobdir to use this functionality.")

        iterator = LogIterator(app_id, log_file, self, should_tail=should_tail)
        # sometimes there's multiple lines per logged line
        iterator = split_lines_iterator(iterator)
        if regex:
            iterator = filter_regex(regex, iterator)
        return iterator

    def list(self) -> List[ListAppResponse]:
        ret = []
        p = subprocess.run(["bjobs", "-noheader", "-a", "-o", "proj stat exit_code"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=True)
        lines = p.stdout.decode("utf-8").split("\n")
        apps = {}
        for line in lines:
            if line != "":
                split = line.split(" ")
                state = get_job_state(split[1], split[2])
                if split[0] not in apps.keys():
                    apps[split[0]] = []
                apps[split[0]].append(state)
        for app_id, states in apps.items():
            success_count = 0
            app_state = AppState.RUNNING
            for state in states:
                if state == AppState.FAILED or state == AppState.CANCELLED or state == AppState.PENDING or state == AppState.UNKNOWN:
                    app_state = state
                    break
                elif state == AppState.SUCCEEDED:
                    success_count += 1
            if success_count == len(states):
                app_state = AppState.SUCCEEDED
            ret.append(ListAppResponse(app_id=app_id, state=app_state))
        return ret

def create_scheduler(session_name: str, **kwargs: Any) -> LsfScheduler:
	return LsfScheduler(
		session_name=session_name,
	)

