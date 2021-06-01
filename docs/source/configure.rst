Configuring
======================

TorchX defines plugin points for you to configure TorchX to best support
your infrastructure setup. Most of the configuration is done through
Python's `entry points<https://packaging.python.org/specifications/entry-points/>`.

The entry points described below can be specified in your project's `setup.py`
file as

.. code-block:: python

 from setuptools import setup

 setup(
    name="project foobar",
    entry_points={
        "$ep_group_name" : [
            "$ep_name = $module:$function"
        ],
    }
 )


Registering Custom Schedulers
--------------------------------
You may implement a custom scheduler by implementing the
.. py::class torchx.schedulers.Scheduler interface. Once you do you can
register this custom scheduler with the following entrypoint

::

 [torchx.schedulers]
 $sched_name = my.custom.scheduler:create_scheduler

The ``create_scheduler`` function should have the following function signature:

.. code-block:: python

 from torchx.schedulers import Scheduler

 def create_scheduler(session_name: str, **kwargs: Any) -> Scheduler:
     return MyScheduler(session_name, **kwargs)


Registering Named Resources
-------------------------------

A Named Resource is a set of predefined resource specs that are given a
string name. This is particularly useful
when your cluster has a fixed set of instance types. For instance if your
deep learning training kubernetes cluster on AWS is
comprised only of p3.16xlarge (64 vcpu, 8 gpu, 488GB), then you may want to
enumerate t-shirt sized resource specs for the containers as:

.. code-block:: python

 {
    "gpu_x_1" : "Resource(cpu=8,  gpu=1, memMB=61*GB),
    "gpu_x_2" : "Resource(cpu=16, gpu=2, memMB=122*GB),
    "gpu_x_4" : "Resource(cpu=32, gpu=4, memMB=244*GB),
    "gpu_x_8" : "Resource(cpu=64, gpu=8, memMB=488*GB),
 }


And refer to the resources by their string names.

<COMING SOON>

Registering Custom Components
-------------------------------
It is possible to author and register a custom set of components with the
``torchx`` CLI as builtins to the CLI. This makes it possible to customize
a set of components most relevant to your team or organization and support
it as a CLI ``builtin``. This way users will see your custom components
when they run

.. code-block:: shell-session

 $ torchx builtins

<COMING SOON>