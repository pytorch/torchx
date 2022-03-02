torchx.specs
======================

.. automodule:: torchx.specs
.. currentmodule:: torchx.specs


AppDef
------------

.. autoclass:: AppDef
   :members:

Role
------------
.. autoclass:: Role
   :members:

.. autoclass:: RetryPolicy
   :members:

Resource
------------
.. autoclass:: Resource
   :members:

.. autofunction:: resource

.. autofunction:: get_named_resources


AWS Named Resources
^^^^^^^^^^^^^^^^^^^^^
.. automodule:: torchx.specs.named_resources_aws

.. currentmodule:: torchx.specs.named_resources_aws

.. autofunction:: aws_m5_2xlarge
.. autofunction:: aws_p3_2xlarge
.. autofunction:: aws_p3_8xlarge
.. autofunction:: aws_t3_medium

Macros
------------
.. currentmodule:: torchx.specs
.. autoclass:: macros
   :members:


Run Configs
--------------
.. autoclass:: runopts
   :members:

Run Status
--------------
.. autoclass:: AppStatus
   :members:

.. autoclass:: AppState
   :members:

.. autoclass:: ReplicaState
   :members:

Component Linter
-----------------
.. automodule:: torchx.specs.file_linter
.. currentmodule:: torchx.specs.file_linter

.. autofunction:: validate
.. autofunction:: get_fn_docstring

.. autoclass:: LinterMessage
   :members:

.. autoclass:: TorchFunctionVisitor
   :members:

.. autoclass:: TorchXArgumentHelpFormatter
   :members:

.. autoclass:: TorchxFunctionArgsValidator
   :members:

.. autoclass:: TorchxFunctionValidator
   :members:

.. autoclass:: TorchxReturnValidator
   :members:
