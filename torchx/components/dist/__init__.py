# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Components for applications that run as distributed jobs. Many of the
components in this section are simply topological, meaning that they define
the layout of the nodes in a distributed setting and take the actual
binaries that each group of nodes (``specs.Role``) runs.
"""
