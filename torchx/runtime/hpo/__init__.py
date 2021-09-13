#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
The ``torchx.runtime.hpo`` module contains modules and functions that you
can use to build a Hyperparameter Optimization (HPO) application. Note
that an HPO application is the entity that is coordinating the HPO search
and is not to be confused with the application that runs for each "trial"
of the search. Typically a "trial" in an HPO job is the trainer app that
trains an ML model given a set of parameters as dictated by the HPO job.

For grid-search, the HPO job may be as simple as a for-parallel loop that
exhaustively runs through all the combinations of parameters in the user-defined
search space. On the other hand, bayesian optimization requires the optimizer
state to be preserved between trials, which leads a more non-trivial implementation
of an HPO app.

Currently this module uses `Ax <https://ax.dev>`_ as the underlying brains of HPO
and offers a few extension points to integrate Ax with TorchX runners.

Quick Usage:

.. code-block:: python

 from ax.modelbridge.dispatch_utils import choose_generation_strategy
 from ax.service.scheduler import SchedulerOptions
 from ax.service.utils.best_point import get_best_parameters
 from ax.service.utils.report_utils import exp_to_df
 from torchx.runtime.hpo.ax import AppMetric, TorchXRunner, TorchXScheduler


"""
