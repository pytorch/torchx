#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import struct


def make_unique(name: str) -> str:
    """
    Appends a unique 64-bit string to the input argument.

    Returns:
        string in format $name-$unique_suffix
    """
    rand_suffix = random_id()
    return f"{name}-{rand_suffix}"


def random_uint64() -> int:
    """
    random_uint64 returns an random unsigned 64 bit int.
    """
    return struct.unpack("!Q", os.urandom(8))[0]


def random_id() -> str:
    """
    Generates an alphanumeric string ID that matches the requirements from
    https://kubernetes.io/docs/concepts/overview/working-with-objects/names/
    """
    START_CANDIDATES = "abcdefghijklmnopqrstuvwxyz"
    END_CANDIDATES = START_CANDIDATES + "012345679"

    out = ""
    v = random_uint64()
    while v > 0:
        if out == "":
            candidates = START_CANDIDATES
        else:
            candidates = END_CANDIDATES

        char = v % len(candidates)
        v = v // len(candidates)
        out += candidates[char]
    return out
