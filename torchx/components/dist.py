# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
For distributed training, TorchX relies on the scheduler's gang scheduling
capabilities to schedule ``n`` copies of nodes. Once launched, the application
is expected to be written in a way that leverages this topology, for instance,
with PyTorch's
`DDP <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_.
You can express a variety of node topologies with TorchX by specifying multiple
:py:class:`torchx.specs.Role` in your component's AppDef. Each role maps to
a homogeneous group of nodes that performs a "role" (function) in the overall
training. Scheduling-wise, TorchX launches each role as a sub-gang.


A DDP-style training job has a single role: trainers. Whereas a
training job that uses parameter servers will have two roles: parameter server, trainer.
You can specify different entrypoint (executable), num replicas, resource requirements,
and more for each role.


DDP Builtin
----------------

DDP-style trainers are common and easy to templetize since they are homogeneous
single role AppDefs, so there is a builtin: ``dist.ddp``. Assuming your DDP
training script is called ``main.py``, launch it as:

.. code:: shell-session

    # locally, 1 node x 4 workers
    $ torchx run -s local_cwd dist.ddp -j 1x4 --script main.py

    # locally, 2 node x 4 workers (8 total)
    $ torchx run -s local_cwd dist.ddp -j 2x4 --script main.py

    # remote (optionally pass --rdzv_port to use a different master port than the default 29500)
    $ torchx run -s kubernetes -cfg queue=default dist.ddp \\
        -j 2x4 \\
        --script main.py \\


Note that the only difference compared to the local launch is the scheduler (``-s``).
The ``dist.ddp`` builtin uses ``torchelastic`` (more specifically ``torch.distributed.run``)
under the hood. Read more about torchelastic `here <https://pytorch.org/docs/stable/elastic/run.html>`_.

Components APIs
-----------------
"""
import os
import shlex
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torchx
import torchx.specs as specs
from torchx.specs import macros


def ddp(
    *script_args: str,
    script: Optional[str] = None,
    m: Optional[str] = None,
    image: str = torchx.IMAGE,
    name: Optional[str] = None,
    h: Optional[str] = None,
    cpu: int = 2,
    gpu: int = 0,
    memMB: int = 1024,
    j: str = "1x2",
    env: Optional[Dict[str, str]] = None,
    max_retries: int = 0,
    rdzv_port: int = 29500,
    mounts: Optional[List[str]] = None,
) -> specs.AppDef:
    """
    Distributed data parallel style application (one role, multi-replica).
    Uses `torch.distributed.run <https://pytorch.org/docs/stable/distributed.elastic.html>`_
    to launch and coordinate PyTorch worker processes. Defaults to using ``c10d`` rendezvous backend
    on rendezvous_endpoint ``$rank_0_host:$rdzv_port``. Note that ``rdzv_port`` parameter is ignored
    when running on single node, and instead we use port 0 which instructs torchelastic to chose
    a free random port on the host.

    Note: (cpu, gpu, memMB) parameters are mutually exclusive with ``h`` (named resource) where
          ``h`` takes precedence if specified for setting resource requirements.
          See `registering named resources <https://pytorch.org/torchx/latest/advanced.html#registering-named-resources>`_.

    Args:
        script_args: arguments to the main module
        script: script or binary to run within the image
        m: the python module path to run
        image: image (e.g. docker)
        name: job name override (uses the script name if not specified)
        cpu: number of cpus per replica
        gpu: number of gpus per replica
        memMB: cpu memory in MB per replica
        h: a registered named resource (if specified takes precedence over cpu, gpu, memMB)
        j: {nnodes}x{nproc_per_node}, for gpu hosts, nproc_per_node must not exceed num gpus
        env: environment varibles to be passed to the run (e.g. ENV1=v1,ENV2=v2,ENV3=v3)
        max_retries: the number of scheduler retries allowed
        rdzv_port: the port on rank0's host to use for hosting the c10d store used for rendezvous.
                   Only takes effect when running multi-node. When running single node, this parameter
                   is ignored and a random free port is chosen.
        mounts: mounts to mount into the worker environment/container (ex. type=<bind/volume>,src=/host,dst=/job[,readonly]).
                See scheduler documentation for more info.
    """

    if (script is None) == (m is None):
        raise ValueError("exactly one of --script and -m must be specified")

    rep = j.split("x")
    if len(rep) == 1:  # num replicas only
        nnodes = 1
        nproc_per_node = int(rep[0])
    elif len(rep) == 2:
        nnodes = int(rep[0])
        nproc_per_node = int(rep[1])
    else:
        raise ValueError(f"Invalid format for -j, usage example: 1x4. Given: {j}")

    if script:
        # script name/module no extension
        role_name = Path(script).stem
    elif m:
        role_name = m.rpartition(".")[2]
    else:
        raise ValueError("failed to compute role_name")

    rdzv_backend = "c10d"
    if nnodes == 1:
        # using port 0 makes elastic chose a free random port which is ok
        # for single-node jobs since all workers run under a single agent
        rdzv_endpoint = "localhost:0"
    else:
        rdzv_endpoint = _noquote(f"$${macros.rank0_env}:{rdzv_port}")

    if env is None:
        env = {}
    env.setdefault("LOGLEVEL", os.getenv("LOGLEVEL", "WARNING"))

    cmd = [
        "python",
        "-m",
        "torch.distributed.run",
        "--rdzv_backend",
        rdzv_backend,
        "--rdzv_endpoint",
        rdzv_endpoint,
        "--rdzv_id",
        f"{macros.app_id}",
        "--nnodes",
        str(nnodes),
        "--nproc_per_node",
        str(nproc_per_node),
        "--tee",
        "3",
        "--role",
        "",
    ]
    if script is not None:
        cmd += [script]
    elif m is not None:
        cmd += ["-m", m]
    cmd += script_args
    return specs.AppDef(
        name=name or role_name,
        roles=[
            specs.Role(
                name=role_name,
                image=image,
                entrypoint="bash",
                num_replicas=nnodes,
                resource=specs.resource(cpu=cpu, gpu=gpu, memMB=memMB, h=h),
                args=["-c", _args_join(cmd)],
                env=env,
                port_map={
                    "c10d": 29500,
                },
                max_retries=max_retries,
                mounts=specs.parse_mounts(mounts) if mounts else [],
            )
        ],
    )


def _args_join(args: Iterable[str]) -> str:
    """
    _args_join is like shlex.join but if the argument is wrapped in _noquote
    it'll not quote that argument.
    """
    quoted = [arg if isinstance(arg, _noquote) else shlex.quote(arg) for arg in args]
    return " ".join(quoted)


class _noquote(str):
    """
    _noquote is a wrapper around str that indicates that the argument shouldn't
    be passed through shlex.quote.
    """

    pass
