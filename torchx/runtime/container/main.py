# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import importlib
import json
import os
import os.path
import sys
from typing import Type, List, Set, Callable, Optional

import torchx
from torchx.runtime.component import is_optional, Component

# pyre-fixme[24]: Generic type `Component` expects 3 type parameters.
def get_component_class(path: str) -> Type[Component]:
    dot = path.rindex(".")
    module_name = path[:dot]
    class_name = path[dot + 1 :]
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    assert issubclass(cls, Component)
    return cls


def _get_parser(field: Type[object]) -> Callable[[str], object]:
    for cls in (int, float, str):
        if cls == field or field == Optional[cls]:
            return cls

    return json.loads


def main(args: List[str]) -> None:
    print(f"torchx version: {torchx.__version__}")
    print(f"process args: {args}")
    component_name = args[1]
    print(f"component_name: {component_name}")

    parser = argparse.ArgumentParser(prog="torchx-main")
    cls = get_component_class(component_name)
    Config, Inputs, Outputs = cls._get_args()
    value_fields: Set[str] = set()
    for arg_cls in (Config, Inputs, Outputs):
        for field, fieldtype in arg_cls.__annotations__.items():
            value_fields.add(field)
            required = not is_optional(fieldtype)
            parser.add_argument(
                f"--{field}", required=required, dest=field, type=_get_parser(fieldtype)
            )
            if arg_cls == Outputs:
                parser.add_argument(
                    f"--output-path-{field}",
                    required=False,
                    dest=f"output_path_{field}",
                )

    parsed = parser.parse_args(args[2:])
    inputs = vars(parsed)

    for field in Outputs.__annotations__.keys():
        if output_path := inputs[f"output_path_{field}"]:
            dirname = os.path.dirname(output_path)
            os.makedirs(dirname, exist_ok=True)
            with open(output_path, "wt") as f:
                val = inputs[field]
                assert isinstance(
                    val, str
                ), f"{field}: output_path only supports str, got {val}"
                f.write(val)

    # pyre-fixme[45]: Cannot instantiate abstract class `Component`.
    component = cls(**inputs)
    component.run(component.inputs, component.outputs)


if __name__ == "__main__":
    main(sys.argv)
