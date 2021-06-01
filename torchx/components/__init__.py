# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This module contains a collection of builtin TorchX components. The directory
structure is organized by component category. Components are simply
templetized app specs. Think of them as a factory methods for different types
of job definitions. The functions that return ``specs.Application`` in this
module are what we refer to as components.

You can browse the library of components on
`github <https://github.com/pytorch/torchx/tree/master/torchx/components>`_
or on our :ref:`docs page<Components Library>`.


"""
