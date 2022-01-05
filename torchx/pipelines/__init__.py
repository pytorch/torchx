# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
These modules are adapters used to run TorchX components as part of a pipeline to allow
for more complex behaviors as well as for continuous deployment.

The runner and schedulers are designed to launch a single component quickly
where as these adapters transform the component into something understandable by
the specific pipeline provider so you can assemble a full pipeline with them.
"""
