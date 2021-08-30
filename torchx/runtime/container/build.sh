#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

gtar -czh --exclude docs --exclude .git . | docker build -t torchx - -f torchx/runtime/container/Dockerfile
