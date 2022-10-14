#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
import struct

START_CANDIDATES: str = "bcdfghjklmnpqrstvwxz"
END_CANDIDATES: str = START_CANDIDATES + "012345679"


def make_unique(name: str, string_length: int = 0) -> str:
    """
    Appends a unique 64-bit string to the input argument.

    Returns:
        string in format $name-$unique_suffix
    """
    return (
        f"{name}-{random_id()}"
        if string_length == 0
        else f"{name}-{get_len_random_id(string_length)}"
    )


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


def get_len_random_id(string_length: int) -> str:
    """
    Generates an alphanumeric string ID that matches the requirements from
    https://kubernetes.io/docs/concepts/overview/working-with-objects/names/
    """
    out = ""
    for i in range(string_length):
        if out == "":
            candidates = START_CANDIDATES
        else:
            candidates = END_CANDIDATES

        out += random.choice(candidates)

    return out
