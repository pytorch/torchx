# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict

INIT_COUNT = 0


def init_plugin(args: Dict[str, str]) -> None:
    global INIT_COUNT
    INIT_COUNT += 1

    assert args["foo"] == "bar", f"invalid arguments passed: {args}"
