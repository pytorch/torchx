Component Best Practices
==========================

This has a list of common things you might want to do with a component and best
practices for them. Components are designed to be flexible so you can deviate
from these practices if necessary however these are the best practices we use
for the builtin TorchX components.

See :ref:`app_best_practices:App Best Practices` for information how to write
apps using TorchX.

Simplify
-------------------

When writing a component generally you want to keep it as simple as possible and
make it so it just passes the arguments around.

Avoiding complex processing makes it easier to use the component in different
environments and easier for others to understand what the component does.

Argument Processing
^^^^^^^^^^^^^^^^^^^^^

Argument processing makes it hard to use the component in other environments. In
this example we concatenate the image name which makes it impossible to use in
other environments which have a different format for images.

.. code-block:: python

   # avoid
   def trainer(img_name: str, img_version: str) -> AppDef:
       """
       ...
       """
       return AppDef(roles=[Role(image=f"{img_name}:{img_version}")...)

   # recommended
   def trainer(image: str):
       """
       ...
       """
       return AppDef(roles=[Role(image=image)...)

Branching Logic
^^^^^^^^^^^^^^^^^

If the component has lots of branching logic it can make it hard for others to
understand how the component maps to the underlying app and how they should
configure the arguments.

.. code-block:: python

   # avoid
   def trainer(stage: str):
       """
       ...
       """
       if stage == "test":
           num_replicas=1
       elif stage == "prod":
           num_replicas=10
       return AppDef(roles=[Role(..., num_replicas=num_replicas)])

   # recommended
   def trainer_test():
       """
       ...
       """
       return trainer(num_replicas=1)

   def trainer_prod() -> AppDef:
       """
       ...
       """
       return trainer(num_replicas=10)

   # not a component just a function
   def trainer(num_replicas: int) -> AppDef:
        return AppDef(roles=[Role(..., num_replicas=num_replicas)])



Named Resources
-----------------

Generally when writing components it's best to use TorchX's named resources
support instead of manually specifying cpu and memory allocations. Named
resources allow your component to be environment independent and allow for
better scheduling behavior by using t-shirt sizes.

See :py:meth:`torchx.specs.get_named_resources` for more info.

Composing Components
----------------------

For common component styles we provide base component definitions. These can be
called from your custom component definition and are generally easier to use
than creating a full AppDef from scratch.

See:

* :py:mod:`torchx.components.base` for simple single node components.
* :py:meth:`torchx.components.dist.ddp` for distributed components.

For even more complex components it's possible to merge multiple existing
components into a single one. For instance you could use a metrics UI component
and merge the roles from it into training component roles to have a sidecar
service to your main training job.

Distributed Components
------------------------

If you're writing a component for distributed training or other similar
distributed computation, we recommend using the
:py:meth:`torchx.components.dist.ddp` component since it provides out of the box
support for `torch.distributed.elastic` jobs.

You can extend the `ddp` component by writing a custom component that simple
imports the `ddp` component and calls it with your app configuration.

Define All Arguments
----------------------

It's generally preferable to define all component arguments as function
arguments instead of consuming a dictionary of arguments. This makes it easier
for users to figure out the options as well as can provide static typing when
used with `pyre <https://pyre-check.org/>`__ or `mypy <http://mypy-lang.org/>`__.

Unit Tests
--------------

.. automodule:: torchx.components.component_test_base
.. currentmodule:: torchx.components.component_test_base

.. autoclass:: torchx.components.component_test_base.ComponentTestCase
   :members:
   :private-members: _validate

Integration Tests
-------------------

You can setup integration tests with your components by either using the
programmatic runner API or write a bash script to call the CLI.

You can see both styles in use in the core
`TorchX scheduler integration tests <https://github.com/pytorch/torchx/tree/main/.github/workflows>`__.
