#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains adapters for converting TorchX components into KubeFlow
Pipeline components.

The current KFP adapters only support single node (1 role and 1 replica)
components.
"""

import kfp

from .version import __version__ as __version__  # noqa F401


def _check_kfp_version() -> None:
    if not kfp.__version__.startswith("1."):
        raise ImportError(
            f"Only kfp version 1.x.x is supported! kfp version {kfp.__version__}"
        )


_check_kfp_version()
