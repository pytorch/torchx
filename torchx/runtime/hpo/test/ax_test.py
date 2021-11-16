#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
import unittest
from typing import List

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


class TorchXSchedulerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp("torchx_runtime_hpo_ax_test")

        self.old_cwd = os.getcwd()
        os.chdir(os.path.dirname(__file__))

        self._parameters: List[Parameter] = [
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

        self._minimize = True
        self._objective = Objective(
            metric=AppMetric(
                name="booth_eval",
            ),
            minimize=self._minimize,
        )

        self._runner = TorchXRunner(
            tracker_base=self.test_dir,
            component=utils.booth,
            scheduler="local_cwd",
            cfg={"prepend_cwd": True},
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)
        os.chdir(self.old_cwd)

    def test_run_experiment_locally(self) -> None:
        # runs optimization over n rounds of k sequential trials

        experiment = Experiment(
            name="torchx_booth_sequential_demo",
            search_space=SearchSpace(parameters=self._parameters),
            optimization_config=OptimizationConfig(objective=self._objective),
            runner=self._runner,
            is_test=True,
            properties={Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF: True},
        )

        # maybe add-on cfg into SchedulerOption?
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

        for i in range(3):
            scheduler.run_n_trials(max_trials=2)

        # print statement intentional for demo purposes
        print(exp_to_df(experiment))

        # AppMetrics always returns trial index; hence the best
        # experiment for min objective will be the params for trial 0
        best_param, _ = none_throws(get_best_parameters(experiment))
        # nothing to assert, just make sure experiment runs

    def test_run_experiment_locally_in_batches(self) -> None:
        # runs optimization over k x n rounds of k parallel trials
        # note:
        #   1. setting max_parallelism_cap in generation_strategy
        #   2. setting run_trials_in_batches in scheduler options
        #   3. setting total_trials = parallelism * rounds
        #
        # this asks ax to run up to max_parallelism_cap trials
        # in parallel by submitting them to the scheduler at the same time

        parallelism = 2
        rounds = 3

        experiment = Experiment(
            name="torchx_booth_parallel_demo",
            search_space=SearchSpace(parameters=self._parameters),
            optimization_config=OptimizationConfig(objective=self._objective),
            runner=self._runner,
            is_test=True,
            properties={Keys.IMMUTABLE_SEARCH_SPACE_AND_OPT_CONF: True},
        )

        # maybe add-on cfg into SchedulerOption?
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
        # nothing to assert, just make sure experiment runs

    def test_runner_no_batch_trials(self) -> None:
        experiment = Experiment(
            name="runner_test",
            search_space=SearchSpace(parameters=self._parameters),
            optimization_config=OptimizationConfig(objective=self._objective),
            runner=self._runner,
            is_test=True,
        )

        with self.assertRaises(ValueError):
            self._runner.run(trial=BatchTrial(experiment))
