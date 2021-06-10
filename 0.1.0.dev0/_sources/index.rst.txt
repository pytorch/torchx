:github_url: https://github.com/pytorch/torchx

TorchX
==================

TorchX is an application spec SDK for PyTorch related apps. It defines
standard structs called ``specs`` that represent the job definition of an
application. The application ``spec`` is the common language between
TorchX ``runners`` and pipeline ``adapters``. Once an application's ``spec``
is created, the application can be run as a standalone job on a cluster or
as a stage in an ML pipeline/workflow. TorchX works with several mainstream
job schedulers and ML pipeline platforms so chances are you are already familiar
with one that works with TorchX.

Just getting started? First learn the :ref:`basic concepts<Basics>` and
take a look at the :ref:`builtin components<Components Library>` library.

Not finding the component or adapter you are looking for? Write a custom one
that fits your needs by using our :ref:`SDK<torchx.api>`.

Documentation
---------------

.. toctree::
   :maxdepth: 1
   :caption: Usage

   basics
   quickstart
   cli
   configure


.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples_apps/index
   examples_pipelines/index


.. _torchx.api:
.. toctree::
   :maxdepth: 1
   :caption: API

   specs
   components
   runner
   schedulers
   pipelines


Components Library
---------------------
.. toctree::
   :maxdepth: 1
   :caption: Components

   components/base
   components/distributed
   components/serve
   components/hpo
   components/utils

Works With
---------------

.. _Schedulers:
.. toctree::
   :maxdepth: 1
   :caption: Schedulers

   schedulers/local
   schedulers/kubernetes
   schedulers/slurm

.. _Pipelines:
.. toctree::
   :maxdepth: 1
   :caption: Pipeline Adapters

   pipelines/kfp

Experimental
---------------
.. toctree::
   :maxdepth: 1
   :caption: Beta Features

   beta



