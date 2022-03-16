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
    # remote (needs you to start a local etcd server on port 2379! and have a `python-etcd` library installed)
    $ torchx run -s local_cwd dist.ddp
        -j 2x4 \\
        --rdzv_endpoint localhost:2379 \\
        --script main.py \\

    # remote (needs you to setup an etcd server first!)
    $ torchx run -s kubernetes -cfg queue=default dist.ddp \\
        -j 2x4 \\
        --script main.py \\


There is a lot happening under the hood so we strongly encourage you
to continue reading the rest of this section
to get an understanding of how everything works. Also note that
while ``dist.ddp`` is convenient, you'll find that authoring your own
distributed component is not only easy (simplest way is to just copy
``dist.ddp``!) but also leads to better flexbility and maintainability
down the road since builtin APIs are subject to more changes than
the more stable specs API. However the choice is yours, feel free to rely on
the builtins if they meet your needs.

Distributed Training
-----------------------

Local Testing
===================

.. note:: Please follow :ref:`examples_apps/lightning_classy_vision/component:Prerequisites of running examples` first.


Running distributed training locally is a quick way to validate your
training script. TorchX's local scheduler will create a process per
replica (``--nodes``). The example below uses `torchelastic <https://pytorch.org/docs/stable/elastic/run.html>`_,
as the main entrypoint of each node, which in turn spawns ``--nprocs_per_node`` number
of trainers. In total you'll see ``nnodes*nprocs_per_node`` trainer processes and ``nnodes``
elastic agent procesess created on your local host.

.. code:: shell-session

   $ torchx run -s local_docker ./torchx/examples/apps/lightning_classy_vision/component.py:trainer_dist \\
        --nnodes 2 \\
        --nproc_per_node 2 \\
        --rdzv_backend c10d \\
        --rdzv_endpoint localhost:29500


Remote Launching
====================

.. note:: Please follow the :ref:`schedulers/kubernetes:Prerequisites` first.

The following example demonstrate launching the same job remotely on kubernetes.

.. code:: shell-session

    $ torchx run -s kubernetes -cfg queue=default \\
        ./torchx/examples/apps/lightning_classy_vision/component.py:trainer_dist \\
        --nnodes 2 \\
        --nproc_per_node 2 \\
        --rdzv_backend etcd \\
        --rdzv_endpoint etcd-server.default.svc.cluster.local:2379
    torchx 2021-10-18 18:46:55 INFO     Launched app: kubernetes://torchx/default:cv-trainer-pa2a7qgee9zng
    torchx 2021-10-18 18:46:55 INFO     AppStatus:
      msg: <NONE>
      num_restarts: -1
      roles: []
      state: PENDING (2)
      structured_error_msg: <NONE>
      ui_url: null

    torchx 2021-10-18 18:46:55 INFO     Job URL: None

Note that the only difference compared to the local launch is the scheduler (``-s``)
and ``--rdzv_backend``. etcd will also work in the local case, but we used ``c10d``
since it does not require additional setup. Note that this is a torchelastic requirement
not TorchX. Read more about rendezvous `here <https://pytorch.org/docs/stable/elastic/rendezvous.html>`_.

.. note:: For GPU training, keep ``nproc_per_node`` equal to the amount of GPUs on the host and
          change the resource requirements in ``torchx/examples/apps/lightning_classy_vision/component.py:trainer_dist``
          method. Modify ``resource_def`` to the number of GPUs that your host has.


Components APIs
-----------------
"""
import os
import shlex
from pathlib import Path
from typing import Dict, Iterable, Optional, List

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
    rdzv_backend: str = "c10d",
    rdzv_endpoint: Optional[str] = None,
    mounts: Optional[List[str]] = None,
) -> specs.AppDef:
    """
    Distributed data parallel style application (one role, multi-replica).
    Uses `torch.distributed.run <https://pytorch.org/docs/stable/distributed.elastic.html>`_
    to launch and coordinate pytorch worker processes.

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
        rdzv_backend: rendezvous backend (only matters when nnodes > 1)
        rdzv_endpoint: rendezvous server endpoint (only matters when nnodes > 1), defaults to rank0 host for schedulers that support it
        mounts: mounts to mount into the worker environment/container (ex. type=<bind/volume>,src=/host,dst=/job[,readonly]). See scheduler documentation for more info.
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

    if rdzv_endpoint is None:
        rdzv_endpoint = _noquote(f"$${macros.rank0_env}:29500")

    if nnodes == 1:
        rdzv_backend = "c10d"
        rdzv_endpoint = "localhost:29500"

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
