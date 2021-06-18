#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
The ``torchx.sdk`` module contains TorchX's base interfaces, objects, and tools
for:

1. Authoring :py:class:`Component`
2. Authoring :py:class:`ComponentAdapter`
3. Authoring various types of plugins


Use this SDK to implement custom functionalities. This is the same SDK
that TorchX uses to implement the out-of-the-box:

1. Components (the ones in ``torchx.apps.*``)
2. Adapters to certain ML Platforms (e.g. ``torchx.kfp`` for Kubeflow Pipelines)
3. Plugins to certain cloud providers (e.g. ``torchx.plugins.aws``)

Refer to the implementations of these out of the box functionalities
as examples when implementing custom functionalities.
"""
