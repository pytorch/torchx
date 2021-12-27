# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Union


def parse_args(arg: str) -> Dict[str, Union[str]]:
    conf = {}
    for kv in arg.split(","):
        if kv == "":
            continue
        key, value = kv.split("=")
        conf[key] = value
    return conf


def parse_as_list(arg: str) -> List[str]:
    conf = []
    for el in arg.split(","):
        conf.append(el)
    return conf


def parse_args_children(arg: str) -> Dict[str, Union[str, List[str]]]:
    conf = {}
    for key, value in parse_args(arg).items():
        if ";" in value:
            value = value.split(";")
        conf[key] = value
    return conf
