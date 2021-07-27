#!/bin/sh
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

WORK_DIR=/tmp/papermill

set -ex
mkdir -p "$WORK_DIR"

# create empty master.tar.gz file and setup symlinks instead of pulling from
# master so we can handle local changes
tar -cJf "$WORK_DIR/master.tar.gz" -T /dev/null
ROOT="$(pwd)/.."
(cd "$WORK_DIR" && ln -s "$ROOT/torchx" . && ln -s "$ROOT/examples" .)

files="$(find "$(pwd)"/build -name '*.ipynb')"
for file in $files
do
  echo "Processing $file..."
  (cd "$WORK_DIR" && papermill "$file" /tmp/papermill-build.ipynb)
done
