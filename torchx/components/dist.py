# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Torchx helps you to run your distributed trainer jobs. Check out :py:mod:`torchx.components.train`
on the example of running single trainer job. Here we will be using
the same :ref:`examples_apps/lightning_classy_vision/train:Trainer App Example`.
but will run it in a distributed manner.

Torchx uses `Torch distributed run <https://pytorch.org/docs/stable/elastic/run.html>`_ to launch user processes
and expects that user applications will be written in
`Distributed data parallel <https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html>`_
manner


Distributed Trainer Execution
------------------------------

In order to run your trainer on a single or multiple processes, remotely or locally, all you need to do is to
write a distributed torchx component. The example that we will be using here is
:ref:`examples_apps/lightning_classy_vision/component:Distributed Trainer Component`

The component defines how the user application is launched and torchx will take care of translating this into
scheduler-specific definitions.

.. note:

    Follow :ref:`examples_apps/lightning_classy_vision/component:Prerequisites of running examples` to
    before running the examples

Single node, multiple trainers (desktop)
=========================================

Try launching a single node, multiple trainers example on your desktop:

.. code:: bash

    cd torchx

    torchx run -s local_cwd \
./torchx/examples/apps/lightning_classy_vision/component.py:trainer_dist --skip-export\
--nnodes 1 --nproc_per_node 2

The `./torchx/examples/apps/lightning_classy_vision/component.py:trainer_dist` is reference to the component
function: :ref:`examples_apps/lightning_classy_vision/component:Distributed Trainer Component`



Single node, multiple trainers (kubernetes)
============================================


Now lets launch the same component on the kubernetes cluster.
Check out :py:mod:`torchx.schedulers.kubernetes_scheduler` on dependencies that needs to be installed
before running using `kubernetes` scheduler.


We can use the following cmd to launch application on kubernetes:

.. code:: bash

    torchx run -s kubernetes --scheduler_args namespace=default,queue=default\
./torchx/examples/apps/lightning_classy_vision/component.py:trainer_dist --skip-export\
--nnodes 1 --nproc_per_node 2

The `namespaces` arg corresponds to the kubernetes namespace that you want to launch.
The `queue` arg is the volcano `queue <https://volcano.sh/en/docs/queue/>`_.


Multiple nodes, multiple trainers (kubernetes)
===============================================

It is simple to launch multiple nodes trainer in kubernetes:

.. code:: bash

    torchx run -s kubernetes --scheduler_args namespace=default,queue=default\
./torchx/examples/apps/lightning_classy_vision/component.py:trainer_dist --skip-export\
--nnodes 2 --nproc_per_node 2



Builtin distributed components
---------------------------------

In the examples above we used custom components to launch user applications. It is not always the case that
users need to write their own components.

Torchx comes with set of builtin component that describe typical execution patterns.


dist.ddp
=========

``dist.ddp``  is a component  for applications that run as distributed jobs in a DDP manner.
You can use it to quickly iterate over your application without the need of authoring your own component.

.. note::

    ``dist.ddp`` is a generic component, as a result it is good for quick iterations, but not production usage.
    It is recommended to author your own component if you want to put your application in production.
    Learn more :ref:`components/overview:Authoring` about how to author your component.

We will be using ``dist.ddp`` to execute the following example:

.. code:: python

    # main.py
    import os

    import torch
    import torch.distributed as dist
    import torch.nn.functional as F
    import torch.distributed.run

    def compute_world_size():
        rank = int(os.getenv("RANK", "0"))
        world_size = int(os.getenv("WORLD_SIZE", "1"))
        dist.init_process_group()
        print("successfully initialized process group")

        t = F.one_hot(torch.tensor(rank), num_classes=world_size)
        dist.all_reduce(t)
        computed_world_size = int(torch.sum(t).item())
        print(
            f"rank: {rank}, actual world_size: {world_size}, computed world_size: {computed_world_size}"
        )

    if __name__ == "__main__":
        compute_world_size()


Single trainer on desktop
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can run this example on desktop on four processes using the following cmd:

.. code:: bash

    torchx run -s local_cwd dist.ddp --entrypoint main.py --nproc_per_node 4


Single trainer on kubernetes cluster
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We can execute it on the kubernetes cluster

.. code:: bash

    torchx run -s kubernetes --scheduler_args namespace=default,queue=default\
dist.ddp --entrypoint main.py --nproc_per_node 4



Components APIs
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
