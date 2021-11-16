# Copyright (c) Facebook, Inc. and its affiliates.
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
from pathlib import Path
from typing import Optional

import torchx
import torchx.specs as specs
from torchx.specs import macros


def ddp(
    *script_args: str,
    script: str,
    image: str = torchx.IMAGE,
    name: Optional[str] = None,
    h: str = "aws_t3.medium",
    j: str = "1x2",
    rdzv_endpoint: str = "etcd-server.default.svc.cluster.local:2379",
) -> specs.AppDef:
    """
    Distributed data parallel style application (one role, multi-replica).
    Uses `torch.distributed.run <https://pytorch.org/docs/stable/distributed.elastic.html>`_
    to launch and coordinate pytorch worker processes.

    Args:
        script_args: arguments to the main module
        script: script or binary to run within the image
        image: image (e.g. docker)
        name: job name override (uses the script name if not specified)
        h: a registered named resource
        j: {nnodes}x{nproc_per_node}, for gpu hosts, nproc_per_node must not exceed num gpus
        rdzv_endpoint: etcd server endpoint (only matters when nnodes > 1)
    """

    rep = j.split("x")
    if len(rep) == 1:  # num replicas only
        nnodes = 1
        nproc_per_node = int(rep[0])
    elif len(rep) == 2:
        nnodes = int(rep[0])
        nproc_per_node = int(rep[1])
    else:
        raise ValueError(f"Invalid format for -j, usage example: 1x4. Given: {j}")

    script_name_noext = Path(script).stem  # script name no extension
    return specs.AppDef(
        name=name or script_name_noext,
        roles=[
            specs.Role(
                name=script_name_noext,
                image=image,
                entrypoint="python",
                num_replicas=nnodes,
                resource=specs.named_resources[h],
                args=[
                    "-m",
                    "torch.distributed.run",
                    "--rdzv_backend",
                    ("c10d" if nnodes == 1 else "etcd"),
                    "--rdzv_endpoint",
                    ("localhost:29500" if nnodes == 1 else rdzv_endpoint),
                    "--rdzv_id",
                    f"{macros.app_id}",
                    "--nnodes",
                    str(nnodes),
                    "--nproc_per_node",
                    str(nproc_per_node),
                    script,
                    *script_args,
                ],
            )
        ],
    )
