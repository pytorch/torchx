# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from .api import AppRun


def app_run_from_env() -> AppRun:
    """
    Syntax sugar for `AppRun.run_from_env` method than can be referenced directly from the module.
    """
    return AppRun.run_from_env()
