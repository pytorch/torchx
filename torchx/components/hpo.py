# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""

TorchX integrates with `Ax <https://github.com/facebook/Ax>`_ to provide
hyperparameter optimization (HPO) support. Since the semantics of an HPO
job is highly customizable, especially in the case of Bayesian optimization,
an HPO application is hard to generalize as an executable. Therefore,
HPO is offered as a runtime module rather than a builtin component. This means
that TorchX provides you with the libraries and tools to simplify the building
of your own HPO application and component.

HPO App
---------

#. See :ref:`HPO with Ax + TorchX<runtime/hpo:Overview & Usage>` to learn how to author an HPO application
#. Build an image (typically a Docker image) that includes your HPO app,
   :ref:`author a component<components/overview:Authoring>`
#. Run it with the :ref:`torchx CLI<cli:CLI>` or :py:mod:`torchx.pipelines`

At a high level, the HPO app sets up the HPO experiment and search space. Each HPO
trial is a job that is defined by the AppDef obtained by evaluating the
TorchX component at a point in the parameter space. This point is determined
by the bayesian optimizer within the Ax platform.

The search space dimensions have to line up with the arguments
of the component that you will be running as trials. To launch the HPO app, you can either run the
main directly or invoke it remotely using TorchX (you'll need to author a component for your HPO
app in this case). The diagram below depicts how this works.

.. image:: hpo_diagram.png

Example
---------

The `ax_test.py <https://github.com/pytorch/torchx/blob/main/torchx/runtime/hpo/test/ax_test.py>`_
unittest is a great end-to-end example of how everything works. It demonstrates running an
HPO experiment where each trial the TorchX builtin component :py:func:`torchx.components.utils.booth`.

"""
