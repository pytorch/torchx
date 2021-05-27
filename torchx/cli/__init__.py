# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
The ``torchx`` CLI is a commandline tool around :py:class:`torchx.runner.Runner`.
It allows users to launch :py:class:`torchx.specs.Application` direcly onto
one of the supported schedulers without authoring a pipeline (aka workflow).
This is convenient for quickly iterating on the application logic without
incurring both the technical and cognitive overhead of learning, writing, and
dealing with pipelines.


**PROTIP:** When in doubt use ``torchx --help``.

Quickstart
-----------------
In TorchX the :py:class:`torchx.specs.Application` dataclass is essentially a
job definition. The ``run`` subcommand takes as input an ``Application`` and
runs it with the supplied scheduler arguments. You provide the ``Application``
to the CLI through a regular python file of the form:

::

  from typing import List, Dict
  import torchx.specs as specs

  def get_app_spec(foo: int, bar: List[str], args: List[str]) -> specs.Application:
    \"""
    My foobar application. Document parameters using google style python
    docstrings (https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)

    Args:
        foo (int): Primitives (int, float, str, bool) are valid
        bar (List[str]): So are 1st order containers (List, Dict) of primitives
        baz (str): defaults are allowed, if none provided are treated as required params
    \"""

    role = specs.Role(name="trainer", entrypoint="trainer_main.py", args=bar
    app = specs.Application(name="foobar_app", roles=[

::

  % torchx run --scheduler local echo --help

Usage
-----------------





::

  % torchx run --scheduler local echo --help

Launching an App
~~~~~~~~~~~~~~~~~~~~~~


"""
