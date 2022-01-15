#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This contains an experimental patching runner that can overlay a workspace on
top of the provided image. This allows for fast iterations without having to
rebuild a new image with your application code.

The workspace is a fsspec filesystem that gets walked and overlaid on the image.
This allows having multiple different interfaces such as from Jupyter Notebooks
as well as local file systems.
"""

import logging
import warnings
from typing import Dict, List, Mapping, Optional

from torchx.runner.api import Runner
from torchx.schedulers import get_schedulers
from torchx.schedulers.api import WorkspaceScheduler
from torchx.specs import (
    AppDef,
    AppDryRunInfo,
    AppHandle,
    CfgVal,
    SchedulerBackend,
    from_function,
)
from torchx.specs.finder import get_component


log: logging.Logger = logging.getLogger(__name__)


class WorkspaceRunner(Runner):
    """
    WorkspaceRunner is a special runner that takes an optional workspace
    argument for the run and dryrun_component methods. If a workspace is
    specified a new image will be built with the workspace overlaid on top.

    WARNING: This is in prototype stage and may have backwards incompatible
    changes made without notice.
    """

    def run_component(
        self,
        component: str,
        component_args: List[str],
        scheduler: SchedulerBackend,
        workspace: Optional[str],
        cfg: Optional[Mapping[str, CfgVal]] = None,
    ) -> AppHandle:
        dryrun_info = self.dryrun_component(
            component,
            component_args,
            scheduler,
            workspace,
            cfg,
        )
        return self.schedule(dryrun_info)

    def dryrun_component(
        self,
        component: str,
        component_args: List[str],
        scheduler: SchedulerBackend,
        workspace: Optional[str],
        cfg: Optional[Mapping[str, CfgVal]] = None,
    ) -> AppDryRunInfo:
        component_def = get_component(component)
        app = from_function(component_def.fn, component_args)
        return self.dryrun(app, scheduler, workspace, cfg)

    def dryrun(
        self,
        app: AppDef,
        scheduler: SchedulerBackend,
        workspace: Optional[str],
        cfg: Optional[Mapping[str, CfgVal]] = None,
    ) -> AppDryRunInfo:
        if workspace:
            self._patch_app(app, scheduler, workspace)

        return super().dryrun(app, scheduler, cfg)

    def _patch_app(self, app: AppDef, scheduler: str, workspace: str) -> None:
        sched = self._scheduler(scheduler)
        if not isinstance(sched, WorkspaceScheduler):
            warnings.warn(
                f"can't apply workspace to image since {sched} is not a "
                "WorkspaceScheduler"
            )
            return

        log.info(f"building patch images for workspace: {workspace}...")

        images = {}
        for role in app.roles:
            img = images.get(role.image)
            if not img:
                img = sched.build_workspace_image(role.image, workspace)
                images[role.image] = img
            role.image = img

    def __enter__(self) -> "WorkspaceRunner":
        return self


def get_workspace_runner(
    name: Optional[str] = None,
    component_defaults: Optional[Dict[str, Dict[str, str]]] = None,
    **scheduler_params: object,
) -> WorkspaceRunner:
    """
    Returns a WorkspaceRunner. See torchx.runner.get_runner for more info.
    """
    if not name:
        name = "torchx"

    schedulers = get_schedulers(session_name=name, **scheduler_params)
    return WorkspaceRunner(
        name=name,
        schedulers=schedulers,
        component_defaults=component_defaults,
    )
