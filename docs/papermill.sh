#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

WORK_DIR=/tmp/papermill

set -e
mkdir -p "$WORK_DIR"
files="$(find "$(pwd)"/build -name '*.ipynb')"
for file in $files
do
  echo "Processing $file..."
  (cd "$WORK_DIR" && papermill "$file" /tmp/papermill-build.ipynb)
done
