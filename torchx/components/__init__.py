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
~~~~~~~~~~~~~~~~~~~~~~~~

Component is a well-defined python function that accepts arguments and returns `torchx.specs.AppDef`.
The examples of components can be found under :py:mod:`torchx.components` module.

Component function has the following properties:

* All arguments of the function must be annotated with corresponding data type
* Component function can use the following types:
    + Primitives: int, float, str, bool
    + Bool is supported via key-value pair, e.g.: `--foo True` correspond to `foo:bool = True`
    + Optional primitives: Optional[int], Optional[float], Optional[str]
    + Dict: Dict[Primitive_key, Primitive_value]
    + List: List[Primitive_value]
    + Optional[List[Primitive_value]], Optional[Dict[Primitive_key, Primitive_value]]
    + VAR_ARG. Component function may define `*arg`, that accepts arbitrary number of arguments
      and can be passed to the underlying script.
* The function should have well defined description in
    https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html format


Torchx supports running builtin component as well as runnig components via provided path:

::

  # component function should contain `foo:str` argument
  torchx run ./my_component.py:component_ft --foo bar


Note about VAR_ARG.
~~~~~~~~~~~~~~~~~~~~~~~~

If component function defines python var arg, there will be the following restrictions when launching it via command line:

* The script args should be passed at the end of the command, e.g.

::

  # component.py
  def my_component(*script_args: str, image: str) -> specs.AppDef:
    ...

  # script_args = [arg1, arg2, arg2], image = foo
  torchx run ./component.py:my_component --image foo arg1 arg2 arg3

  # script_args = [--flag, arg1], image = foo
  torchx run ./component.py:my_component --image foo -- --flag arg1

  # script_args = [arg1, --flag], image = foo
  torchx run ./component.py:my_component --image foo arg1 --flag

  # script_args = [--help], image = foo
  torchx run ./component.py:my_component --image foo -- --help

  # script_args = [--image, bar], image = foo
  torchx run ./component.py:my_component --image foo -- --image bar


"""
