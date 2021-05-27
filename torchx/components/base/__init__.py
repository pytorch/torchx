# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
These components are meant to be used as convenience methods when constructing
other components. Many methods in the base component library are factory methods
for ``Role``, ``Container``, and ``Resources`` that are hooked up to
TorchX's configurable extension points.
"""

import torchx.specs as specs


def named_resource(name: str) -> specs.Resource:
    # TODO <PLACEHOLDER> read instance types and resource mappings from entrypoints
    return specs.NULL_RESOURCE
