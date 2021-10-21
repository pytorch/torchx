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
    $ torchx run -s local_cwd dist.ddp --entrypoint main.py --nproc_per_node 4

    # locally, 2 node x 4 workers (8 total)
    $ torchx run -s local_cwd dist.ddp --entrypoint main.py \\
        --rdzv_backend c10d \\
        --nnodes 2 \\
        --nproc_per_node 4 \\

    # remote (needs you to setup an etcd server first!)
    $ torchx run -s kubernetes -cfg queue=default dist.ddp \\
        --entrypoint main.py \\
        --rdzv_backend etcd \\
        --rdzv_endpoint etcd-server.default.svc.cluster.local:2379 \\
        --nnodes 2 \\
        --nproc_per_node 4 \\


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

   $ torchx run -s local_cwd ./torchx/examples/apps/lightning_classy_vision/component.py:trainer_dist \\
        --nnodes 2 \\
        --nproc_per_node 2 \\
        --rdzv_backend c10d \\
        --rdzv_endpoint localhost:29500


.. warning:: There is a known issue with ``local_docker`` (the default scheduler when no ``-s``
             argument is supplied), hence we use ``-s local_cwd`` instead. Please track
             the progress of the fix on `issue-286 <https://github.com/pytorch/torchx/issues/286>`_,
             `issue-287 <https://github.com/pytorch/torchx/issues/287>`_.


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

from typing import Any, Dict, Optional

import torchx.specs as specs
from torchx.components.base import torch_dist_role
from torchx.version import TORCHX_IMAGE


def ddp(
    *script_args: str,
    entrypoint: str,
    image: str = TORCHX_IMAGE,
    rdzv_backend: Optional[str] = None,
    rdzv_endpoint: Optional[str] = None,
    resource: Optional[str] = None,
    nnodes: int = 1,
    nproc_per_node: int = 1,
    name: str = "test-name",
    role: str = "worker",
    env: Optional[Dict[str, str]] = None,
) -> specs.AppDef:
    """
    Distributed data parallel style application (one role, multi-replica).
    Uses `torch.distributed.run <https://pytorch.org/docs/stable/distributed.elastic.html>`_
    to launch and coordinate pytorch worker processes.

    Args:
        script_args: Script arguments.
        image: container image.
        entrypoint: script or binary to run within the image.
        rdzv_backend: rendezvous backend to use, allowed values can be found in the
             `rdzv registry docs <https://github.com/pytorch/pytorch/blob/master/torch/distributed/elastic/rendezvous/registry.py>`_
             The default backend is `c10d`
        rdzv_endpoint: Controller endpoint. In case of rdzv_backend is etcd, this is a etcd
            endpoint, in case of c10d, this is the endpoint of one of the hosts.
            The default entdpoint it `localhost:29500`
        resource: Optional named resource identifier. The resource parameter
            gets ignored when running on the local scheduler.
        nnodes: Number of nodes.
        nproc_per_node: Number of processes per node.
        name: Name of the application.
        role: Name of the ddp role.
        env: Env variables.

    Returns:
        specs.AppDef: Torchx AppDef
    """

    launch_kwargs: Dict[str, Any] = {
        "nnodes": nnodes,
        "nproc_per_node": nproc_per_node,
        "max_restarts": 0,
    }
    if rdzv_backend:
        launch_kwargs["rdzv_backend"] = rdzv_backend
    if rdzv_endpoint:
        launch_kwargs["rdzv_endpoint"] = rdzv_endpoint

    retry_policy: specs.RetryPolicy = specs.RetryPolicy.APPLICATION

    ddp_role = torch_dist_role(
        name=role,
        image=image,
        entrypoint=entrypoint,
        resource=resource or specs.NULL_RESOURCE,
        args=list(script_args),
        env=env,
        num_replicas=nnodes,
        max_retries=0,
        retry_policy=retry_policy,
        port_map=None,
        **launch_kwargs,
    )

    return specs.AppDef(name, roles=[ddp_role])
