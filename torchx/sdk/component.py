#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
from typing import Generic, TypeVar, get_args, Union, Type, Set, Tuple


def is_optional(field: Type[object]) -> bool:
    return Union[field, None] == field


Config = TypeVar("Config")
Inputs = TypeVar("Inputs")
Outputs = TypeVar("Outputs")


class Component(abc.ABC, Generic[Config, Inputs, Outputs]):
    # types (override)
    Version: str

    inputs: Inputs
    outputs: Outputs
    config: Config

    @classmethod
    def _get_args(cls) -> Tuple[Type[Config], Type[Inputs], Type[Outputs]]:
        # pyre-fixme[7]: Expected `Tuple[Type[Variable[Config]],
        #  Type[Variable[Inputs]], Type[Variable[Outputs]]]` but got `Tuple[Any, ...]`.
        # pyre-fixme[16]: `Component` has no attribute `__orig_bases__`.
        return get_args(cls.__orig_bases__[0])

    def __init__(self, **kwargs: object) -> None:
        Config, Inputs, Outputs = self._get_args()

        fields: Set[str] = set()

        self.config = Config()
        self.inputs = Inputs()
        self.outputs = Outputs()

        for paramtype, param in [
            (Config, self.config),
            (Inputs, self.inputs),
            (Outputs, self.outputs),
        ]:
            for field, fieldtype in paramtype.__annotations__.items():
                if field in fields:
                    raise TypeError(
                        f"duplicate field name {field} in definition of {self.__class__}"
                    )
                fields.add(field)

                v = kwargs.get(field)
                if not is_optional(fieldtype) and v is None:
                    raise TypeError(f"missing required argument {field}")

                # pyre-fixme[16]: `Config` has no attribute `__setitem__`.
                param[field] = v

    @abc.abstractmethod
    def run(self, inputs: Inputs, outputs: Outputs) -> None:
        raise NotImplementedError
