# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import inspect
import unittest
from typing import Dict, List, Union, Optional, cast

import typing_inspect
from torchx.util.types import (
    decode_from_string,
    decode_optional,
    is_primitive,
    is_bool,
    to_dict,
    to_list,
)


def _test_complex_args(
    arg1: int, arg2: Optional[List[str]], arg3: Union[float, int]
) -> int:
    return 42


def _test_dict(arg1: Dict[int, float]) -> int:
    return 42


def _test_list(arg1: List[float]) -> int:
    return 42


def _test_complex_list(arg1: List[List[float]]) -> int:
    return 42


def _test_unknown(arg1: Optional[Optional[float]]) -> int:
    return 42


class TypesTest(unittest.TestCase):
    def test_decode_optional(self) -> None:
        parameters = inspect.signature(_test_complex_args).parameters

        arg1_parameter = parameters["arg1"]
        arg1_type = decode_optional(arg1_parameter.annotation)
        self.assertTrue(arg1_type is int)

        arg2_parameter = parameters["arg2"]
        arg2_type = decode_optional(parameters["arg2"].annotation)
        self.assertTrue(typing_inspect.get_origin(arg2_type) is list)

        arg3_parameter = parameters["arg3"]
        arg3_type = decode_optional(arg3_parameter.annotation)
        self.assertTrue(typing_inspect.get_origin(arg3_type) is Union)

    def test_is_primitive(self) -> None:
        parameters = inspect.signature(_test_complex_args).parameters

        arg1_parameter = parameters["arg1"]
        arg1_type = decode_optional(arg1_parameter.annotation)
        self.assertTrue(is_primitive(arg1_parameter.annotation))

        arg2_parameter = parameters["arg2"]
        arg2_type = decode_optional(parameters["arg2"].annotation)
        self.assertFalse(is_primitive(arg2_parameter.annotation))

    def test_is_bool(self) -> None:
        self.assertTrue(is_bool(bool))
        self.assertFalse(is_bool(int))

    def test_decode_from_string_dict(self) -> None:
        parameters = inspect.signature(_test_dict).parameters

        encoded_value = "1=1.0,2=42.1,3=10"

        value = decode_from_string(encoded_value, parameters["arg1"].annotation)
        value = cast(Dict[int, float], value)
        self.assertEqual(3, len(value))
        self.assertEqual(float(1.0), value[1])
        self.assertEqual(float(42.1), value[2])
        self.assertEqual(float(10), value[3])

    def test_decode_from_string_list(self) -> None:
        parameters = inspect.signature(_test_list).parameters

        encoded_value = "1.0,42.2,3.9"

        value = decode_from_string(encoded_value, parameters["arg1"].annotation)
        value = cast(List[float], value)
        self.assertEqual(3, len(value))
        self.assertEqual(float(1.0), value[0])
        self.assertEqual(float(42.2), value[1])
        self.assertEqual(float(3.9), value[2])

    def test_decode_from_string_empty(self) -> None:
        parameters = inspect.signature(_test_list).parameters
        value = decode_from_string("", parameters["arg1"].annotation)
        self.assertEqual(None, value)

    def test_decode_from_string_complex_list(self) -> None:
        parameters = inspect.signature(_test_complex_list).parameters
        with self.assertRaises(ValueError):
            decode_from_string("1", parameters["arg1"].annotation)

    def test_decode_from_string_unknown(self) -> None:
        parameters = inspect.signature(_test_unknown).parameters
        with self.assertRaises(ValueError):
            decode_from_string("2", parameters["arg1"].annotation)

    def test_to_dict_empty(self) -> None:
        self.assertDictEqual({}, to_dict(""))

    def test_to_dict_simple(self) -> None:
        enc = "foo=bar,key=value"
        self.assertDictEqual({"foo": "bar", "key": "value"}, to_dict(enc))

    def test_to_dict_one(self) -> None:
        enc = "foo=bar"
        self.assertDictEqual({"foo": "bar"}, to_dict(enc))

    def test_to_dict_only_key(self) -> None:
        enc = "foo"
        with self.assertRaises(ValueError):
            to_dict(enc)

    def test_to_dict_complex_comma(self) -> None:
        enc = "foo=bar1,bar2,bar3,my_key=my_value,new_foo=new_bar1,new_bar2"
        self.assertDictEqual(
            {
                "foo": "bar1,bar2,bar3",
                "my_key": "my_value",
                "new_foo": "new_bar1,new_bar2",
            },
            to_dict(enc),
        )

    def test_to_dict_doulbe_comma(self) -> None:
        enc = "key1=value1,,foo=bar"
        self.assertDictEqual({"foo": "bar", "key1": "value1,"}, to_dict(enc))

    def test_to_list_empty(self) -> None:
        self.assertListEqual([], to_list(""))
