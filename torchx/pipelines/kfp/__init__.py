#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

"""
This module contains adapters for converting TorchX components into KubeFlow
Pipeline components.

The current KFP adapters only support single node (1 role and 1 replica)
components.
"""

import warnings

import kfp

from .version import __version__ as __version__  # noqa F401


def _check_kfp_version() -> None:
    if kfp.__version__.startswith("1."):
        warnings.warn(
            f"KFP version 1.x.x is deprecated! Please upgrade to kfp version 2.x.x. Current version: {kfp.__version__}",
            DeprecationWarning,
        )


_check_kfp_version()
