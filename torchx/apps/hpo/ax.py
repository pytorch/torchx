#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Callable, Dict, Optional, Set, cast

import pandas as pd
from ax.core import Trial
from ax.core.abstract_data import AbstractDataFrameData
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.metric import Metric
from ax.core.runner import Runner as ax_Runner
from ax.service.scheduler import Scheduler as ax_Scheduler, TrialStatus
from pyre_extensions import none_throws
from torchx.runner import Runner, get_runner
from torchx.specs import AppDef, AppState, AppStatus, RunConfig


_TORCHX_APP_HANDLE: str = "torchx_app_handle"
_TORCHX_RUNNER: str = "torchx_runner"

# maps torchx AppState to Ax's TrialStatus equivalent
APP_STATE_TO_TRIAL_STATUS: Dict[AppState, TrialStatus] = {
    AppState.UNSUBMITTED: TrialStatus.CANDIDATE,
    AppState.SUBMITTED: TrialStatus.STAGED,
    AppState.PENDING: TrialStatus.STAGED,
    AppState.RUNNING: TrialStatus.RUNNING,
    AppState.SUCCEEDED: TrialStatus.COMPLETED,
    AppState.CANCELLED: TrialStatus.ABANDONED,
    AppState.FAILED: TrialStatus.FAILED,
    AppState.UNKNOWN: TrialStatus.FAILED,
}


class AppMetric(Metric):
    def fetch_trial_data(
        self, trial: BaseTrial, **kwargs: Any
    ) -> AbstractDataFrameData:
        # TODO implement
        # sketch of idea:
        #  1. Have the job take --trial_result_path = s3://foo/bar
        #     (the argument name can be whatever, we just care about the uri)
        #  2. Read the uri
        #  3. return the resulting metrics wrapped in ax.core.data.Data
        #
        # for now just return the trial idx

        # no need to check for trial type since we checked already in TorchXRunner
        # and refused to run anythin other than trials of type `Trial`

        df_dict = {
            "arm_name": none_throws(cast(Trial, trial).arm).name,
            "trial_index": trial.index,
            "metric_name": self.name,  # FIXME get this from the output file
            "mean": trial.index,  # FIXME get this from the output file
            "sem": None,
        }
        return Data(df=pd.DataFrame.from_records([df_dict]))


class TorchXRunner(ax_Runner):
    """
    An implementation of ``ax.core.runner.Runner`` that delegates job submission
    to the TorchX Runner. This runner is coupled with the torchx component
    since Ax runners run trials of a single component with different parameters.

    It is expected that the experiment parameter names and types match
    EXACTLY with component function args.

    For example for the component shown below:

    .. code-block:: python

     def trainer_component(x1: int, x2: float) -> spec.AppDef:
        # ... implementation omitted for brevity ...
        pass

    The experiment should be set up as:

    .. code-block:: python

     parameters=[
       {
         "name": "x1",
         "value_type": "int",
         # ... other options...
       },
       {
         "name": "x2",
         "value_type": "float",
         # ... other options...
       }
     ]

    """

    def __init__(
        self,
        component: Callable[..., AppDef],
        scheduler: str,
        scheduler_args: Optional[RunConfig] = None,
        torchx_runner: Optional[Runner] = None,
    ) -> None:
        self._component: Callable[..., AppDef] = component
        self._scheduler: str = scheduler
        self._scheduler_args: Optional[RunConfig] = scheduler_args
        # need to use the same runner in case it has state
        # e.g. torchx's local_scheduler has state hence need to poll status
        # on the same scheduler instance
        self._torchx_runner: Runner = torchx_runner or get_runner()

    def run(self, trial: BaseTrial) -> Dict[str, Any]:
        """
        Submits the trial (which maps to an AppDef) as a job
        onto the scheduler using ``torchx.runner``.

        ..  note:: only supports `Trial` (not `BatchTrial`).
        """

        if not isinstance(trial, Trial):
            raise ValueError(
                f"{type(trial)} is not supported. Check your experiment setup"
            )

        # TODO add type + param check here
        parameters = none_throws(trial.arm).parameters or {}
        appdef = self._component(**parameters)

        app_handle = self._torchx_runner.run(
            appdef, self._scheduler, self._scheduler_args
        )
        return {_TORCHX_APP_HANDLE: app_handle, _TORCHX_RUNNER: self._torchx_runner}


class TorchXScheduler(ax_Scheduler):
    """
    An implementation of an `Ax Scheduler<https://ax.dev/tutorials/scheduler.html>`_
    that works with Experiments hooked up with the ``TorchXRunner``.

    This scheduler is not a real scheduler but rather a facade scheduler
    that delegates to scheduler clients for various remote/local schedulers.
    For a list of supported schedulers please refer to TorchX
    `scheduler docs<https://pytorch.org/torchx/latest/schedulers.html>`_.

    """

    def poll_trial_status(self) -> Dict[TrialStatus, Set[int]]:
        trial_statuses: Dict[TrialStatus, Set[int]] = {}

        for trial in self.running_trials:
            app_handle: str = trial.run_metadata[_TORCHX_APP_HANDLE]
            torchx_runner = trial.run_metadata[_TORCHX_RUNNER]
            app_status: AppStatus = torchx_runner.status(app_handle)
            trial_status = APP_STATE_TO_TRIAL_STATUS[app_status.state]

            indices = trial_statuses.setdefault(trial_status, set())
            indices.add(trial.index)

        return trial_statuses

    def poll_available_capacity(self) -> Optional[int]:
        """
        Used when ``run_trials_in_batches`` option is set.
        Since this scheduler is a faux scheduler, this method
        always returns the ``max_parallelism`` of the current
        step of this scheduler's ``generation_strategy``.

        .. note:: The trials (jobs) are simply submitted to the
                  scheduler in parallel. Typically the trials will be
                  queued in the scheduler's job queue (on the server-side)
                  and executed according to the scheduler's job priority
                  and scheduling policies.

        """

        return self.generation_strategy._curr.max_parallelism
