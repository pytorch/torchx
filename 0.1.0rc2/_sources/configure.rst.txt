Configuring
======================

TorchX defines plugin points for you to configure TorchX to best support
your infrastructure setup. Most of the configuration is done through
Python's `entry points <https://packaging.python.org/specifications/entry-points/>`__.

.. note::

  Entry points requires a python package containing them be installed.
  If you don't have a python package we recommend you make one so you can share
  your resource definitions, schedulers and components across your team and org.

The entry points described below can be specified in your project's `setup.py`
file as

.. testsetup:: setup

   import sys
   sys.argv = ["setup.py", "--version"]

.. testcode:: setup

 from setuptools import setup

 setup(
     name="project foobar",
     entry_points={
         "torchx.schedulers": [
             "my_scheduler = my.custom.scheduler:create_scheduler",
         ],
         "torchx.named_resources": [
             "gpu_x2 = my_module.resources:gpu_x2",
         ],
     }
 )

.. testoutput:: setup
   :hide:

   0.0.0



Registering Custom Schedulers
--------------------------------
You may implement a custom scheduler by implementing the
.. py::class torchx.schedulers.Scheduler interface.

The ``create_scheduler`` function should have the following function signature:

.. testcode::

 from torchx.schedulers import Scheduler

 def create_scheduler(session_name: str, **kwargs: object) -> Scheduler:
     return MyScheduler(session_name, **kwargs)

You can then register this custom scheduler by adding an entry_points definition
to your python project.


.. testcode::

   # setup.py
   ...
   entry_points={
       "torchx.schedulers": [
           "my_scheduler = my.custom.scheduler:create_schedule",
       ],
   }



Registering Named Resources
-------------------------------

A Named Resource is a set of predefined resource specs that are given a
string name. This is particularly useful
when your cluster has a fixed set of instance types. For instance if your
deep learning training kubernetes cluster on AWS is
comprised only of p3.16xlarge (64 vcpu, 8 gpu, 488GB), then you may want to
enumerate t-shirt sized resource specs for the containers as:

.. testcode:: python

 from torchx.specs import Resource

 def gpu_x1() -> Resource:
     return Resource(cpu=8,  gpu=1, memMB=61_000)

 def gpu_x2() -> Resource:
     return Resource(cpu=16, gpu=2, memMB=122_000)

 def gpu_x3() -> Resource:
     return Resource(cpu=32, gpu=4, memMB=244_000)

 def gpu_x4() -> Resource:
     return Resource(cpu=64, gpu=8, memMB=488_000)

.. testcode:: python
 :hide:

 gpu_x1()
 gpu_x2()
 gpu_x3()
 gpu_x4()

To make these resource definitions available you then need to register them via
entry_points:

.. testcode::

   # setup.py
   ...
   entry_points={
       "torchx.named_resources": [
           "gpu_x2 = my_module.resources:gpu_x2",
       ],
   }


Once you install the package with the entry_points definitions, the named
resource can then be used in the following manner:

.. testsetup:: role

   from torchx.specs import named_resources, Resource

   named_resources["gpu_x2"] = Resource(cpu=16, gpu=2, memMB=122_000)


.. doctest:: role

   >>> from torchx.specs import get_named_resources
   >>> get_named_resources("gpu_x2")
   Resource(cpu=16, gpu=2, memMB=122000, ...)


.. testcode:: role

  # my_module.component
  from torchx.specs import AppDef, Role, get_named_resources

  def test_app(resource: str) -> AppDef:
      return AppDef(name="test_app", roles=[
          Role(
              name="...",
              image="...",
              resource=get_named_resources(resource),
          )
      ])

  test_app("gpu_x2")


Registering Custom Components
-------------------------------
It is possible to author and register a custom set of components with the
``torchx`` CLI as builtins to the CLI. This makes it possible to customize
a set of components most relevant to your team or organization and support
it as a CLI ``builtin``. This way users will see your custom components
when they run

.. code-block:: shell-session

 $ torchx builtins

Custom components can be registered via the following modification of the ``entry_points``:


.. testcode::

   # setup.py
   ...
   entry_points={
       "torchx.components": [
           "foo = my_project.bar",
       ],
   }

The line above registers a group ``foo`` that is associated with the module ``my_project.bar``.
Torchx will recursively traverse lowest level dir associated with the ``my_project.bar`` and will find
all defined components.

.. note:: If there are two registry entries, e.g. ``foo = my_project.bar`` and ``test = my_project``
          there will be two sets of overlapping components with different aliases.


After registration, torchx cli will display registered components via:

.. code-block:: shell-session

 $ torchx builtins

If ``my_project.bar`` had the following directory structure:

::

 $PROJECT_ROOT/my_project/bar/
     |- baz.py

And `baz.py` defines a component (function) called `trainer`. Then the component can be run as a job in the following manner:

.. code-block:: shell-session

 $ torchx run foo.baz.trainer -- --name "test app"
