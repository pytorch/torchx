torchx.runner
==============

The runner lets you run components as a standalone job on one of the supported
:ref:`schedulers<Schedulers>`. The runner takes an ``specs.AppDef`` object, which
is the result of evaluating the component function with a set of user-provided
arguments, along with the scheduler name and scheduler args (aka ``runcfg`` or ``runopts``)
and submits the component as a job (see diagram below).

.. image:: runner_diagram.png

Runner Functions
~~~~~~~~~~~~~~~~~~

.. automodule:: torchx.runner
.. currentmodule:: torchx.runner


.. autofunction:: get_runner

Runner Classes
~~~~~~~~~~~~~~~~

.. autoclass:: Runner
   :members:

