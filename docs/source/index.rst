:github_url: https://github.com/pytorch/torchx

TorchX
==================

TorchX is a component SDK and Library for PyTorch ML pipelines.

TorchX lets you build an ML pipeline by composing TorchX
:ref:`provided components<Components>` into a workflow. TorchX components
can be used as pipeline stages in various existing ML platforms by passing them
through one of the :ref:`supported adapters<Adapters>`.

Not finding the component or adapter you are looking for? Write a custom one
that fits your needs by using our :ref:`SDK<Documentation>`.

.. image:: components_diagram.jpg
.. note:: Diagram above for illustration purposes only. Not all boxes
          are currently available out-of-the-box.


Get Started
---------------
.. toctree::
   :maxdepth: 1
   :caption: Usage

   quickstart
   configure
   examples


Documentation
---------------

.. toctree::
   :maxdepth: 1
   :caption: SDK

   sdk

Components
---------------

.. toctree::
   :maxdepth: 1
   :caption: Components

   components/data
   components/train
   components/deploy
   components/hpo

Adapters
---------------

.. toctree::
   :maxdepth: 1
   :caption: Adapters

   adapters/kfp

Plugins
---------------
.. toctree::
   :maxdepth: 1
   :caption: Plugins

   plugins/aws
   plugins/azure
   plugins/gcp

Experimental
---------------
.. toctree::
   :maxdepth: 1
   :caption: Beta Features

   beta



