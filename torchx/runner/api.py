#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import getpass
import importlib
import json
import time
from dataclasses import asdict
from datetime import datetime
from pprint import pformat
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pyre_extensions import none_throws
from torchx.runner.events import log_event
from torchx.schedulers import get_schedulers
from torchx.schedulers.api import Scheduler
from torchx.specs.api import (
    AppDef,
    AppDryRunInfo,
    AppHandle,
    AppStatus,
    RunConfig,
    SchedulerBackend,
    UnknownAppException,
    from_file,
    from_function,
    from_module,
    make_app_handle,
    parse_app_handle,
    runopts,
)
from torchx.util import entrypoints


NONE: str = "<NONE>"


class Runner:
    """
    Torchx individual component runner. Has the methods for the user to
    act upon ``AppDefs``. The ``Runner`` is stateful and
    represents a logical workspace of the user. It can be backed by a service
    (e.g. Torchx server) for persistence or can be standalone with no persistence
    meaning that the ``Runner`` lasts only during the duration of the hosting
    process (see the ``attach()`` API for instructions on re-parenting apps
    between sessions).
    """

    def __init__(
        self,
        name: str,
        schedulers: Dict[SchedulerBackend, Scheduler],
        wait_interval: int = 10,
    ) -> None:
        if "default" not in schedulers:
            raise ValueError(
                f"A default scheduler is required. Provided schedulers: {schedulers.keys()}"
            )
        self._name: str = name
        self._schedulers = schedulers
        self._wait_interval = wait_interval
        self._apps: Dict[AppHandle, AppDef] = {}

    def run_from_path(
        self,
        component_path: str,
        app_args: List[str],
        scheduler: SchedulerBackend = "default",
        cfg: Optional[RunConfig] = None,
        dryrun: bool = False,
    ) -> AppHandle:
        """Resolves and runs the application in the specified mode.

        Retrieves application based on ``component_path`` and runs it in the specified mode.

        The ``component_path`` has the following resolution order(from high-pri to low-pri):
            * User-registered components. Users can register components via
                https://packaging.python.org/specifications/entry-points/. Method looks for
                entrypoints in the group ``torchx.components``.
            * File-based components in format: ``$FILE_PATH:FUNCTION_NAME``. Both relative and
                absolute paths supported.
            * Builtin components relative to `torchx.components`. The path to the component should
                be module name relative to `torchx.components` and function name in a format:
                ``$module.$function``.

        Usage:

        ::

         runner.run_from_path("distributed.ddp", ...) - will be resolved to
            ``torchx.components.distributed`` module and ``ddp`` function.


         runner.run_from_path("~/home/components.py:my_component", ...) - will be resolved to
            ``~/home/components.py`` file and ``my_component`` function.


        Returns:
            An application handle that is used to call other action APIs on the app, or ``<NONE>``
            if it dryrun specified.

        Raises:
            AppNotReRunnableException: if the session/scheduler does not support re-running attached apps
            ValueError: if the ``component_path`` is failed to resolve.
        """
        app_fn = entrypoints.load("torchx.components", component_path, default=NONE)
        if app_fn != NONE:
            app = from_function(app_fn, app_args)
        elif ":" in component_path:
            file_path, function_name = component_path.split(":")
            if not function_name:
                raise ValueError(
                    f"Incorrect path: {component_path}. Missing function name after `:`"
                )
            app = from_file(file_path, function_name, app_args)
        else:
            function_name = component_path.split(".")[-1]
            component_module = component_path[
                0 : len(component_path) - len(function_name) - 1
            ]
            full_module = f"torchx.components.{component_module}"
            try:
                app_module = importlib.import_module(full_module)
            except ModuleNotFoundError:
                raise ValueError(
                    f"Incorrect component path: {component_path}. Valid values: \n"
                    "* $file_path:function_name, Example: ~/home/file.py:my_component \n"
                    "* Module and function name to the builtin components, e.g. : distributed.ddp"
                )
            app = from_module(app_module, function_name, app_args)

        if dryrun:
            print("=== APPLICATION ===")
            print(pformat(asdict(app), indent=2, width=80))

            print("=== SCHEDULER REQUEST ===")
            print(self.dryrun(app, scheduler, cfg))
            return "<NONE>"
        else:
            return self.run(app, scheduler, cfg)

    def run(
        self,
        app: AppDef,
        scheduler: SchedulerBackend = "default",
        cfg: Optional[RunConfig] = None,
    ) -> AppHandle:
        """
        Runs the given application in the specified mode.

        .. note:: sub-classes of ``Session`` should implement ``schedule`` method
                  rather than overriding this method directly.

        Returns:
            An application handle that is used to call other action APIs on the app.

        Raises:
            AppNotReRunnableException: if the session/scheduler does not support re-running attached apps
        """

        dryrun_info = self.dryrun(app, scheduler, cfg)
        return self.schedule(dryrun_info)

    # pyre-fixme[24]: AppDryRunInfo was designed to work with Any request object
    def schedule(self, dryrun_info: AppDryRunInfo) -> AppHandle:
        """
        Actually runs the application from the given dryrun info.
        Useful when one needs to overwrite a parameter in the scheduler
        request that is not configurable from one of the object APIs.

        .. warning:: Use sparingly since abusing this method to overwrite
                     many parameters in the raw scheduler request may
                     lead to your usage of TorchX going out of compliance
                     in the long term. This method is intended to
                     unblock the user from experimenting with certain
                     scheduler-specific features in the short term without
                     having to wait until TorchX exposes scheduler features
                     in its APIs.

        .. note:: It is recommended that sub-classes of ``Session`` implement
                  this method instead of directly implementing the ``run`` method.

        Usage:

        ::

         dryrun_info = session.dryrun(app, scheduler="default", cfg)

         # overwrite parameter "foo" to "bar"
         dryrun_info.request.foo = "bar"

         app_handle = session.submit(dryrun_info)

        """
        scheduler_backend = none_throws(dryrun_info._scheduler)
        cfg = dryrun_info._cfg
        runcfg = json.dumps(cfg.cfgs) if cfg else None
        with log_event("schedule", scheduler_backend, runcfg=runcfg) as logger_context:
            sched = self._scheduler(scheduler_backend)
            app_id = sched.schedule(dryrun_info)
            app_handle = make_app_handle(scheduler_backend, self._name, app_id)
            app = none_throws(dryrun_info._app)
            self._apps[app_handle] = app
            _, _, app_id = parse_app_handle(app_handle)
            logger_context._torchx_event.app_id = app_id
            return app_handle

    def name(self) -> str:
        return self._name

    def dryrun(
        self,
        app: AppDef,
        scheduler: SchedulerBackend = "default",
        cfg: Optional[RunConfig] = None,
        # pyre-fixme[24]: AppDryRunInfo was designed to work with Any request object
    ) -> AppDryRunInfo:
        """
        Dry runs an app on the given scheduler with the provided run configs.
        Does not actually submit the app but rather returns what would have been
        submitted. The returned ``AppDryRunInfo`` is pretty formatted and can
        be printed or logged directly.

        Usage:

        ::

         dryrun_info = session.dryrun(app, scheduler="local", cfg)
         print(dryrun_info)

        """
        # input validation
        if not app.roles:
            raise ValueError(
                f"No roles for app: {app.name}. Did you forget to call app.of(roles..)?"
            )

        for role in app.roles:
            if not role.entrypoint:
                raise ValueError(
                    f"No entrypoint for role: {role.name}."
                    f" Did you forget to call role.runs(entrypoint, args, env)?"
                )
            if role.num_replicas <= 0:
                raise ValueError(
                    f"Non-positive replicas for role: {role.name}."
                    f" Did you forget to set role.num_replicas?"
                )
        sched = self._scheduler(scheduler)
        sched._validate(app, scheduler)
        dryrun_info = sched.submit_dryrun(app, cfg or RunConfig())
        dryrun_info._scheduler = scheduler
        return dryrun_info

    def run_opts(self) -> Dict[str, runopts]:
        """
        Returns the ``runopts`` for the supported scheduler backends.

        Usage:

        ::

         local_runopts = session.run_opts()["local"]
         print("local scheduler run options: {local_runopts}")

        Returns:
            A map of scheduler backend to its ``runopts``
        """
        return {
            scheduler_backend: scheduler.run_opts()
            for scheduler_backend, scheduler in self._schedulers.items()
        }

    def scheduler_backends(self) -> List[SchedulerBackend]:
        """
        Returns a list of all supported scheduler backends.
        All session implementations must support a "default"
        scheduler backend and document what the default
        scheduler is.
        """
        return list(self._schedulers.keys())

    def status(self, app_handle: AppHandle) -> Optional[AppStatus]:
        """
        Returns:
            The status of the application, or ``None`` if the app does not exist anymore
            (e.g. was stopped in the past and removed from the scheduler's backend).
        """
        scheduler, scheduler_backend, app_id = self._scheduler_app_id(
            app_handle, check_session=False
        )
        with log_event("status", scheduler_backend, app_id):
            desc = scheduler.describe(app_id)
            if not desc:
                # app does not exist on the scheduler
                # remove it from apps cache if it exists
                # effectively removes this app from the list() API
                self._apps.pop(app_handle, None)
                return None

            app_status = AppStatus(
                desc.state,
                desc.num_restarts,
                msg=desc.msg,
                structured_error_msg=desc.structured_error_msg,
                roles=desc.roles_statuses,
            )
            if app_status:
                app_status.ui_url = desc.ui_url
            return app_status

    def wait(self, app_handle: AppHandle) -> Optional[AppStatus]:
        """
        Block waits (indefinitely) for the application to complete.
        Possible implementation:

        ::

         while(True):
             app_status = status(app)
             if app_status.is_terminal():
                 return
             sleep(10)

        Returns:
            The terminal status of the application, or ``None`` if the app does not exist anymore
        """
        scheduler, scheduler_backend, app_id = self._scheduler_app_id(
            app_handle, check_session=False
        )
        with log_event("wait", scheduler_backend, app_id):
            while True:
                app_status = self.status(app_handle)

                if not app_status:
                    return None
                if app_status.is_terminal():
                    return app_status
                else:
                    time.sleep(self._wait_interval)

    def list(self) -> Dict[AppHandle, AppDef]:
        """
        Returns the applications that were run with this session mapped by the app handle.
        The persistence of the session is implementation dependent.
        """
        with log_event("list"):
            app_ids = list(self._apps.keys())
            for app_id in app_ids:
                self.status(app_id)
            return self._apps

    def stop(self, app_handle: AppHandle) -> None:
        """
        Stops the application, effectively directing the scheduler to cancel
        the job. Does nothing if the app does not exist.

        .. note:: This method returns as soon as the cancel request has been
                  submitted to the scheduler. The application will be in a
                  ``RUNNING`` state until the scheduler actually terminates
                  the job. If the scheduler successfully interrupts the job
                  and terminates it the final state will be ``CANCELLED``
                  otherwise it will be ``FAILED``.

        """
        scheduler, scheduler_backend, app_id = self._scheduler_app_id(app_handle)
        with log_event("stop", scheduler_backend, app_id):
            status = self.status(app_handle)
            if status is not None and not status.is_terminal():
                scheduler.cancel(app_id)

    def describe(self, app_handle: AppHandle) -> Optional[AppDef]:
        """
        Reconstructs the application (to the best extent) given the app handle.
        Note that the reconstructed application may not be the complete app as
        it was submitted via the run API. How much of the app can be reconstructed
        is scheduler dependent.

        Returns:
            AppDef or None if the app does not exist anymore or if the
            scheduler does not support describing the app handle
        """
        scheduler, scheduler_backend, app_id = self._scheduler_app_id(
            app_handle, check_session=False
        )

        with log_event("describe", scheduler_backend, app_id):
            # if the app is in the apps list, then short circuit everything and return it
            app = self._apps.get(app_handle, None)
            if not app:
                desc = scheduler.describe(app_id)
                if desc:
                    app = AppDef(name=app_id).of(*desc.roles)
            return app

    def log_lines(
        self,
        app_handle: AppHandle,
        role_name: str,
        k: int = 0,
        regex: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        should_tail: bool = False,
    ) -> Iterable[str]:
        """
        Returns an iterator over the log lines of the specified job container.

        .. note:: #. ``k`` is the node (host) id NOT the ``rank``.
                  #. ``since`` and ``until`` need not always be honored (depends on scheduler).

        .. warning:: The semantics and guarantees of the returned iterator is highly
                     scheduler dependent. See ``torchx.specs.api.Scheduler.log_iter``
                     for the high-level semantics of this log iterator. For this reason
                     it is HIGHLY DISCOURAGED to use this method for generating output
                     to pass to downstream functions/dependencies. This method
                     DOES NOT guarantee that 100% of the log lines are returned.
                     It is totally valid for this method to return no or partial log lines
                     if the scheduler has already totally or partially purged log records
                     for the application.

        Usage:

        ::

         app_handle = session.run(app, scheduler="local", cfg=RunConfig())

         print("== trainer node 0 logs ==")
         for line in session.log_lines(app_handle, "trainer", k=0):
            print(line)

        Discouraged anti-pattern:

        ::

         # DO NOT DO THIS!
         # parses accuracy metric from log and reports it for this experiment run
         accuracy = -1
         for line in session.log_lines(app_handle, "trainer", k=0):
            if matches_regex(line, "final model_accuracy:[0-9]*"):
                accuracy = parse_accuracy(line)
                break
         report(experiment_name, accuracy)

        Args:
            app_handle: application handle
            role_name: role within the app (e.g. trainer)
            k: k-th replica of the role to fetch the logs for
            regex: optional regex filter, returns all lines if left empty
            since: datetime based start cursor. If left empty begins from the
                    first log line (start of job).
            until: datetime based end cursor. If left empty, follows the log output
                    until the job completes and all log lines have been consumed.

        Returns:
             An iterator over the role k-th replica of the specified application.

        Raise:
            UnknownAppException: if the app does not exist in the scheduler

        """
        scheduler, scheduler_backend, app_id = self._scheduler_app_id(
            app_handle, check_session=False
        )
        with log_event("log_lines", scheduler_backend, app_id):
            if not self.status(app_handle):
                raise UnknownAppException(app_handle)
            log_iter = scheduler.log_iter(
                app_id, role_name, k, regex, since, until, should_tail
            )
            return log_iter

    def _scheduler(self, scheduler: SchedulerBackend) -> Scheduler:
        sched = self._schedulers.get(scheduler)
        if not sched:
            raise KeyError(
                f"Undefined scheduler backend: {scheduler}. Use one of: {self._schedulers.keys()}"
            )
        return sched

    def _scheduler_app_id(
        self, app_handle: AppHandle, check_session: bool = True
    ) -> Tuple[Scheduler, str, str]:
        """
        Returns the scheduler and app_id from the app_handle.
        Set ``check_session`` to validate that the session name in the app handle
        is the same as this session.

        Raises:
            ValueError - if ``check_session=True`` and the session in the app handle
                         does not match this session's name
            KeyError - if no such scheduler backend exists
        """

        scheduler_backend, _, app_id = parse_app_handle(app_handle)
        scheduler = self._scheduler(scheduler_backend)
        return scheduler, scheduler_backend, app_id


def get_runner(name: Optional[str] = None, **scheduler_params: Any) -> Runner:
    """
    Convenience method to construct and get a Runner object.
    """
    if not name:
        name = f"torchx_{getpass.getuser()}"

    schedulers = get_schedulers(session_name=name, **scheduler_params)
    return Runner(name, schedulers)
