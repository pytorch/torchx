#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import boto3

from .session import AwsSessionProvider


def get_session(region: str) -> boto3.Session:
    return AwsSessionProvider().get_session(region)


try:
    # @manual
    from .static_init import *  # noqa: F401 F403
except ModuleNotFoundError:
    pass
