#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""


"""

from torchx.runner import get_runner

if __name__ == "__main__":
    runner = get_runner()
    print(f"Got runner: {runner}")
