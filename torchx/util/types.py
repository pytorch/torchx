# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
from typing import Any, Callable, Dict, List, Type, Union

import typing_inspect


def to_dict(arg: str) -> Dict[str, str]:
    """
    Parses the sting into the key-value pairs using `,` as delimiter.
    The algorithm uses the last `,` between previous value and next key as delimiter, e.g.:

    .. code-block:: python

     arg="FOO=Value1,Value2,BAR=Value3"
     conf = to_dict(arg)
     print(conf) # {'FOO': 'Value1,Value2', 'BAR': 'Value3'}

    """
    conf: Dict[str, str] = {}
    arg = arg.strip()
    if len(arg) == 0:
        return {}

    kv_delimiter: str = "="
    pair_delimiter: str = ","

    cpos: int = 0
    while cpos < len(arg):
        key = _get_key(arg, cpos, kv_delimiter)
        cpos += len(key) + 1
        value = _get_value(arg, cpos, kv_delimiter, pair_delimiter)
        cpos += len(value) + 1
        conf[key] = value
    return conf


def _get_key(arg: str, spos: int, kv_delimiter: str = "=") -> str:
    epos: int = spos + 1
    while epos < len(arg) and arg[epos] != kv_delimiter:
        epos += 1
    if epos == len(arg):
        raise ValueError(
            f"Argument `{arg}` does not follow the pattern: KEY1=VALUE1,KEY2=VALUE2"
        )
    return arg[spos:epos]


def _get_value(arg: str, spos: int, kv_delimiter="=", pair_delimiter=",") -> str:
    epos: int = spos + 1
    while epos < len(arg) and arg[epos] != kv_delimiter:
        epos += 1
    if epos < len(arg) and arg[epos] == kv_delimiter:
        while arg[epos] != pair_delimiter:
            epos -= 1
    return arg[spos:epos]


def to_list(arg: str) -> List[str]:
    conf = []
    if len(arg.strip()) == 0:
        return []
    for el in arg.split(","):
        conf.append(el)
    return conf


# pyre-ignore-all-errors[3, 2]
def _decode_string_to_dict(
    encoded_value: str, param_type: Type[Dict[Any, Any]]
) -> Dict[Any, Any]:
    key_type, value_type = typing_inspect.get_args(param_type)
    arg_values = {}
    for key, value in to_dict(encoded_value).items():
        arg_values[key_type(key)] = value_type(value)
    return arg_values


def _decode_string_to_list(
    encoded_value: str, param_type: Type[List[Any]]
) -> List[Any]:
    value_type = typing_inspect.get_args(param_type)[0]
    if not is_primitive(value_type):
        raise ValueError("List types support only primitives: int, str, float")
    arg_values = []
    for value in to_list(encoded_value):
        arg_values.append(value_type(value))
    return arg_values


def decode_from_string(
    encoded_value: str, annotation: Any
) -> Union[Dict[Any, Any], List[Any], None]:
    """Decodes string representation to the underlying type(Dict or List)

    Given a string representation of the value, the method decodes it according
    to the ``annotation`` type:

    If ``annotation`` is list, the expected format of ``encoded_value`` is:
    value1,value2,..

    If ``annotation`` is dict, the expected format of ``encoded_value`` is:
    key1=value,key2=value2

    Args:
        encoded_value: String encoded complex type, if empty value provided, method
            will return None
        annotation: Complex type. Currently supported list or dict

    Returns:
        List or Dict filled with decoded ``encoded_value``. The type of
        elements in list or dict will be the same as the one in ``annotation``
    """

    if not encoded_value:
        return None
    value_type = annotation
    value_origin = typing_inspect.get_origin(value_type)
    if value_origin is dict:
        return _decode_string_to_dict(encoded_value, value_type)
    elif value_origin is list:
        return _decode_string_to_list(encoded_value, value_type)
    else:
        raise ValueError("Unknown")


def is_bool(param_type: Any) -> bool:
    """Check if the ``param_type`` is bool"""
    return param_type == bool


def is_primitive(param_type: Any) -> bool:
    """Check if the ``param_type`` belongs to a list of
    (int, str, float)

    Args:
        param_type: Parameter type

    Returns:
        True if ``param_type`` is int, str or float, otherwise False
    """
    TYPES = (int, str, float)
    return param_type in TYPES


def decode_optional(param_type: Any) -> Any:
    """Returns underlying type if optional

    Args:
        param_type: Parameter type

    Returns:
        If ``param_type`` is type Optional[INNER_TYPE], method returns INNER_TYPE
        Otherwise returns ``param_type``
    """
    param_origin = typing_inspect.get_origin(param_type)
    if param_origin is not Union:
        return param_type
    key_type, value_type = typing_inspect.get_args(param_type)
    if value_type is type(None):
        return key_type
    else:
        return param_type


def get_argparse_param_type(parameter: inspect.Parameter) -> Callable[[str], object]:
    if is_primitive(parameter.annotation):
        return parameter.annotation
    else:
        return str
