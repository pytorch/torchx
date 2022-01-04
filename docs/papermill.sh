#!/bin/sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

WORK_DIR=/tmp/papermill

rm -r "$WORK_DIR" || true
mkdir -p "$WORK_DIR"

# create empty main.tar.gz file and setup symlinks instead of pulling from
# main so we can handle local changes
tar -cJf "$WORK_DIR/main.tar.gz" -T /dev/null
ROOT="$(pwd)/.."
(cd "$WORK_DIR" && ln -s "$ROOT/torchx" . && ln -s "$ROOT/examples" .)

files="$(find "$(pwd)"/build -wholename '**/_downloads/*.ipynb')"
for file in $files
do
  echo "Processing $file..."
  (cd "$WORK_DIR" && papermill "$file" /tmp/papermill-build.ipynb)
done
