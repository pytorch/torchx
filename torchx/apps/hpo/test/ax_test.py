#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import List, cast

from ax.core import (
    BatchTrial,
    ChoiceParameter,
    Experiment,
    FixedParameter,
    Objective,
    OptimizationConfig,
    Parameter,
    ParameterType,
    SearchSpace,
    Trial,
)
from ax.modelbridge.dispatch_utils import choose_generation_strategy
from ax.service.scheduler import SchedulerOptions
from ax.service.utils.best_point import get_best_parameters
from ax.service.utils.report_utils import exp_to_df
from pyre_extensions import none_throws
from torchx.apps.hpo.ax import AppMetric, TorchXRunner, TorchXScheduler
from torchx.components import utils


def is_ci() -> bool:
    import os

    return os.getenv("GITHUB_ACTIONS") is not None


@unittest.skipIf(is_ci(), "re-enable when a new version of ax-platform releases")
class TorchXSchedulerTest(unittest.TestCase):
    def setUp(self) -> None:
        self._parameters: List[Parameter] = [
            ChoiceParameter(
                name="msg",
                parameter_type=ParameterType.STRING,
                # generate enough search dimensions for the # of trials
                values=[f"hello_{i}" for i in range(30)],
            ),
            FixedParameter(
                name="num_replicas",
                parameter_type=ParameterType.INT,
                value=1,
            ),
        ]

        self._minimize = True
        self._objective = Objective(
            metric=AppMetric(name="foobar"), minimize=self._minimize
        )

        self._runner = TorchXRunner(component=utils.echo, scheduler="local")

    def test_run_experiment_locally(self) -> None:
        # runs optimization over n rounds of k sequential trials

        experiment = Experiment(
            name="torchx_echo_sequential_demo",
            search_space=SearchSpace(parameters=self._parameters),
            optimization_config=OptimizationConfig(objective=self._objective),
            runner=self._runner,
            is_test=True,
        )

        # maybe add-on RunConfig into SchedulerOption?
        # so that we can pass it from one place
        scheduler = TorchXScheduler(
            experiment=experiment,
            generation_strategy=(
                choose_generation_strategy(
                    search_space=experiment.search_space,
                )
            ),
            options=SchedulerOptions(),
        )

        for i in range(2):
            scheduler.run_n_trials(max_trials=3)

        # print statement intentional for demo purposes
        print(exp_to_df(experiment))

        # AppMetrics always returns trial index; hence the best
        # experiment for min objective will be the params for trial 0
        best_param, _ = none_throws(get_best_parameters(experiment))

        self.assertEqual(
            none_throws(cast(Trial, experiment.trials[0]).arm).parameters, best_param
        )

    def test_run_experiment_locally_in_batches(self) -> None:
        # runs optimization over k x n rounds of k parallel trials
        # note:
        #   1. setting max_parallelism_cap in generation_strategy
        #   2. setting run_trials_in_batches in scheduler options
        #   3. setting total_trials = parallelism * rounds
        #
        # this asks ax to run up to max_parallelism_cap trials
        # in parallel by submitting them to the scheduler at the same time

        parallelism = 3
        rounds = 2

        experiment = Experiment(
            name="torchx_echo_parallel_demo",
            search_space=SearchSpace(parameters=self._parameters),
            optimization_config=OptimizationConfig(objective=self._objective),
            runner=self._runner,
            is_test=True,
        )

        # maybe add-on RunConfig into SchedulerOption?
        # so that we can pass it from one place
        scheduler = TorchXScheduler(
            experiment=experiment,
            generation_strategy=(
                choose_generation_strategy(
                    search_space=experiment.search_space,
                    max_parallelism_cap=parallelism,
                )
            ),
            options=SchedulerOptions(
                run_trials_in_batches=True, total_trials=(parallelism * rounds)
            ),
        )

        scheduler.run_all_trials()
        # print statement intentional for demo purposes
        print(exp_to_df(experiment))

        # AppMetrics always returns trial index; hence the best
        # experiment for min objective will be the params for trial 0
        best_param, _ = none_throws(get_best_parameters(experiment))

        self.assertEqual(
            none_throws(cast(Trial, experiment.trials[0]).arm).parameters,
            best_param,
        )

    def test_runner_no_batch_trials(self) -> None:
        experiment = Experiment(
            name="runner_test",
            search_space=SearchSpace(parameters=self._parameters),
            optimization_config=OptimizationConfig(objective=self._objective),
            runner=self._runner,
            is_test=True,
        )

        runner = TorchXRunner(component=utils.echo, scheduler="local")

        with self.assertRaises(ValueError):
            runner.run(trial=BatchTrial(experiment))
