#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

torchx runopts gcp_batch

APP_ID="$(torchx run --wait --scheduler gcp_batch dist.ddp -j 2x2 --max_retries 3 --script torchx/components/integration_tests/test/dummy_app.py)"
torchx status "$APP_ID"

torchx list -s gcp_batch
LIST_LINES="$(torchx list -s gcp_batch | grep -c "$APP_ID")"
if [ "$LIST_LINES" -ne 1 ]
then
    echo "expected $APP_ID to be listed"
    exit 1
fi

torchx log "$APP_ID"
EXPECTED_MSG="hi from main"
LINES="$(torchx log "$APP_ID" | grep -c "$EXPECTED_MSG")"
if [ "$LINES" -ne 2 ]
then
    echo "expected 2 log lines with msg $EXPECTED_MSG"
    exit 1
fi
