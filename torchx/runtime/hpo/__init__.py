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

Quickstart Example
~~~~~~~~~~~~~~~~~~~~

The following example demonstrates running an HPO job on a TorchX component.
We use the builtin ``utils.booth`` component which simply runs an application
that evaluates the booth function at ``(x1, x2)``. The objective is to
find ``x1`` and ``x2`` that minimizes the booth function.

.. testsetup:: [hpo_ax_demo]

 import tempfile
 tmpdir = tempfile.mkdtemp("torchx_runtime_hpo_demo")

.. testcleanup:: [hpo_ax_demo]

 import shutil
 shutil.rmtree(tmpdir)

.. doctest:: [hpo_ax_demo]

 import os
 from ax.core import (
    BatchTrial,
    Experiment,
    Objective,
    OptimizationConfig,
    Parameter,
    ParameterType,
    RangeParameter,
    SearchSpace,
 )
 from ax.modelbridge.dispatch_utils import choose_generation_strategy
 from ax.service.scheduler import SchedulerOptions
 from ax.service.utils.best_point import get_best_parameters
 from ax.service.utils.report_utils import exp_to_df
 from ax.utils.common.constants import Keys
 from pyre_extensions import none_throws
 from torchx.components import utils
 from torchx.runtime.hpo.ax import AppMetric, TorchXRunner, TorchXScheduler
 from torchx.specs import RunConfig

 # Run HPO on the booth function (https://en.wikipedia.org/wiki/Test_functions_for_optimization)

 parameters = [
    RangeParameter(
        name="x1",
        lower=-10.0,
        upper=10.0,
        parameter_type=ParameterType.FLOAT,
    ),
    RangeParameter(
        name="x2",
        lower=-10.0,
        upper=10.0,
        parameter_type=ParameterType.FLOAT,
    ),
 ]

 objective = Objective(metric=AppMetric(name="booth_eval"), minimize=True)

 runner = TorchXRunner(
    tracker_base=tmpdir,
    component=utils.booth,
    component_const_params={
        "image": "ghcr.io/pytorch/torchx:0.1.0rc0",
    },
    scheduler="local", # can also be [kubernetes, slurm, etc]
    scheduler_args=RunConfig({"log_dir": tmpdir, "image_type": "docker"}),
 )

 experiment = Experiment(
     name="torchx_booth_sequential_demo",
     search_space=SearchSpace(parameters=parameters),
     optimization_config=OptimizationConfig(objective=objective),
     runner=runner,
     is_test=True,
     properties={Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF: True},
 )

 scheduler = TorchXScheduler(
     experiment=experiment,
     generation_strategy=(
         choose_generation_strategy(
             search_space=experiment.search_space,
         )
     ),
     options=SchedulerOptions(),
 )


 for i in range(3): # doctest: +SKIP
    scheduler.run_n_trials(max_trials=2) # doctest: +SKIP

 print(exp_to_df(experiment)) # doctest:+ SKIP

"""
