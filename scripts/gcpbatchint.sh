#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

torchx runopts gcp_batch

APP_ID="$(torchx run --wait --scheduler gcp_batch utils.echo --msg hello)"
torchx status "$APP_ID"

torchx list -s gcp_batch
LIST_LINES="$(torchx list -s gcp_batch | grep -c "$APP_ID")"
if [ "$LIST_LINES" -ne 1 ]
then
    echo "expected $APP_ID to be listed"
    exit 1
fi
