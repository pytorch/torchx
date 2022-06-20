torchx.schedulers
====================

TorchX Schedulers define plugins to existing schedulers. Used with the
:ref:`runner<runner:torchx.runner>`, they submit components as jobs onto
the respective scheduler backends. TorchX supports a few :ref:`schedulers<Schedulers>`
out-of-the-box. You can add your own by implementing .. py:class::torchx.schedulers
and :ref:`registering<advanced:Registering Custom Schedulers>` it in the entrypoint.

.. image:: scheduler_diagram.png

All Schedulers
~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 1
   :glob:

   schedulers/*

.. fbcode::

   .. toctree::
      :maxdepth: 1
      :glob:

      schedulers/fb/*

Scheduler Functions
~~~~~~~~~~~~~~~~~~~~

.. automodule:: torchx.schedulers
.. currentmodule:: torchx.schedulers

.. autofunction:: get_schedulers
.. autofunction:: get_scheduler_factories
.. autofunction:: get_default_scheduler_name

Scheduler Classes
~~~~~~~~~~~~~~~~~~~
.. autoclass:: Scheduler
   :members:

.. autoclass:: SchedulerFactory
   :members:
