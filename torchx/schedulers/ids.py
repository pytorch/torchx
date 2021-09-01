#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import binascii
import os


def make_unique(name: str) -> str:
    """
    Appends the unique 64-bit hex string to the input argument.

    Returns:
        string in format $name_$unique_suffix
    """
    rand_suffix = binascii.b2a_hex(os.urandom(8)).decode("utf-8")
    return f"{name}_{rand_suffix}"
