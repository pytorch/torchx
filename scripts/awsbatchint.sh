#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

APP_ID="$(torchx run --wait --scheduler aws_batch -c queue=torchx utils.echo)"
torchx status "$APP_ID"
torchx describe "$APP_ID"
torchx log "$APP_ID"
LINES="$(torchx log "$APP_ID" | wc -l)"

if [ "$LINES" -ne 1 ]
then
    echo "expected 1 log lines"
    exit 1
fi
