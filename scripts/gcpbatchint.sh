#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

# torchx runopts gcp_batch

# APP_ID="$(torchx run --wait --scheduler gcp_batch utils.echo --msg hello)"
# torchx status "$APP_ID"

# torchx list -s gcp_batch
# LIST_LINES="$(torchx list -s gcp_batch | grep -c "$APP_ID")"
# if [ "$LIST_LINES" -ne 1 ]
# then
#     echo "expected $APP_ID to be listed"
#     exit 1
# fi

# torchx log "$APP_ID"
# LINES="$(torchx log "$APP_ID" | grep -c 'hello')"
# if [ "$LINES" -ne 1 ]
# then
#     echo "expected 1 log line with 'hello'"
#     exit 1
# fi


# torchx run --wait --scheduler gcp_batch dist.ddp -j 2x2 --max_retries 3 --script torchx/components/integration_tests/test/dummy_app.py



JOB="$USER-$(uuidgen)"
DIR="/tmp/$JOB"

function cleanup {
  rm  -r "$DIR"
}
trap cleanup EXIT

mkdir "$DIR"
cd "$DIR"

cat <<EOT > main.py
print("hello world!")
EOT

RUN_ARGS="--scheduler gcp_batch dist.ddp -j 2x1 --max_retries 3 --script main.py"


# shellcheck disable=SC2086
APP_ID="$(torchx run --wait $RUN_ARGS)"
torchx status "$APP_ID"
# torchx describe "$APP_ID"
torchx log "$APP_ID"
LINES="$(torchx log "$APP_ID" | grep -c 'hello world')"

if [ "$LINES" -ne 2 ]
then
    echo "expected 2 log lines"
    exit 1
fi

torchx list -s aws_batch
LIST_LINES="$(torchx list -s aws_batch | grep -c "$APP_ID")"

if [ "$LIST_LINES" -ne 1 ]
then
    echo "expected $APP_ID to be listed"
    exit 1
fi
