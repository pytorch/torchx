# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Type, Union

import typing_inspect


def to_dict(arg: str) -> Dict[str, str]:
    conf = {}
    if len(arg.strip()) == 0:
        return {}
    for kv in arg.split(","):
        if kv == "":
            continue
        key, value = kv.split("=")
        conf[key] = value
    return conf


def to_list(arg: str) -> List[str]:
    conf = []
    if len(arg.strip()) == 0:
        return []
    for el in arg.split(","):
        conf.append(el)
    return conf


# pyre-ignore-all-errors[3, 2]
def decode_string_to_dict(
    encoded_value: str, param_type: Type[Dict[Any, Any]]
) -> Dict[Any, Any]:
    key_type, value_type = typing_inspect.get_args(param_type)
    arg_values = {}
    for key, value in to_dict(encoded_value).items():
        arg_values[key_type(key)] = value_type(value)
    return arg_values


def decode_string_to_list(encoded_value: str, param_type: Type[List[Any]]) -> List[Any]:
    value_type = typing_inspect.get_args(param_type)[0]
    arg_values = []
    for value in to_list(encoded_value):
        arg_values.append(value_type(value))
    return arg_values


def decode_from_string(
    encoded_value: str, annotation: Any
) -> Union[Dict[Any, Any], List[Any], None]:
    if not encoded_value:
        return None
    value_type = annotation
    if value_type.__origin__ == Union:
        value_type = annotation.__args__[0]
    if value_type.__origin__ is dict:
        return decode_string_to_dict(encoded_value, value_type)
    elif value_type.__origin__ is list:
        return decode_string_to_list(encoded_value, value_type)
    else:
        raise ValueError("Unknown")


def is_primitive(param_type: Any) -> bool:
    TYPES = (int, str, float)
    return param_type in TYPES
