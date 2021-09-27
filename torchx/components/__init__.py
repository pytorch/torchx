# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains a collection of builtin TorchX components. The directory
structure is organized by component category. Components are simply
templetized app specs. Think of them as a factory methods for different types
of job definitions. The functions that return ``specs.AppDef`` in this
module are what we refer to as components.

You can browse the library of components in the ``torchx.components`` module
or on our :ref:`docs page<index:Components Library>`.

Components can be used out of the box by either torchx cli or torchx sdk.

::

  # using via sdk
  from torchx.runner import get_runner
  get_runner().run_component("distributed.ddp", app_args=[], scheduler="local_cwd", ...)

  # using via torchx-cli

  >> torchx run --scheduler local_cwd distributed.ddp --param1 --param2


Components development

The addition of a new component is pretty straightforward and consists
 of the following steps:

* Determine component location
* Create component as function
* Unit tests

Determine component location. Each component belongs to one or another category,
and should be located accordingly. E.g. the definition of distributed components
should be located in ``distributed.py`` file.

Create component as function. Each component represents a function that accepts
 arbitrary arguments and returns ``specs.AppDef``.

The function should
have the following properties:

* All arguments of the function must be annotated
* Current supported types:
    + Primitives: int, float, str
    + Optional primitives: Optional[int], Optional[float], Optional[str]
    + Dict: Dict[Primitive_key, Primitive_value]
    + List: List[Primitive_value]
    + Optional[List], Optional[Dict]
* The function should have well defined description in
    https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html format


Unit tests. Write unit tests that use ``torchx.specs.file_linter`` to validate
the component's structure,  similar to ``torchx.components.tests.distributed_test.py``.

"""
