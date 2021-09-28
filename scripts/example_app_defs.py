#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
App Defs for integration tests.
"""


import examples.apps.datapreproc.component as dp_component
import examples.apps.dist_cifar.component as dist_cifar_component
import examples.apps.lightning_classy_vision.component as cv_component
from torchx.components.integration_tests.component_provider import ComponentProvider
from torchx.specs import AppDef


class CvTrainerComponentProvider(ComponentProvider):
    def get_app_def(self) -> AppDef:
        return cv_component.trainer(
            image=self._image,
            output_path="/tmp",
            skip_export=True,
            log_path="/tmp",
        )


class DistCifarComponentProvider(ComponentProvider):
    def get_app_def(self) -> AppDef:
        args = ["--output_path", "/tmp", "--dryrun"]
        return dist_cifar_component.trainer(
            *args,
            image=self._image,
        )


class DatapreprocComponentProvider(ComponentProvider):
    def get_app_def(self) -> AppDef:
        return dp_component.data_preproc(
            image=self._image,
            output_path="/tmp/test",
        )
