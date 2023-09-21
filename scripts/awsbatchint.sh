#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

JOB="$USER-$(uuidgen)"
DIR="/tmp/$JOB"

function cleanup {
  rm  -r "$DIR"
}
trap cleanup EXIT

mkdir "$DIR"
cd "$DIR"

cat <<EOT > .torchxconfig
[aws_batch]
queue=torchx
image_repo=495572122715.dkr.ecr.us-west-1.amazonaws.com/torchx/integration-tests
EOT

cat <<EOT > main.py
print("hello world!")
EOT

RUN_ARGS="--scheduler aws_batch dist.ddp -j 2x1 --script main.py"

if [ -z "$AWS_ROLE_ARN" ]; then
  # only dryrun if no secrets
  # shellcheck disable=SC2086
  torchx run --dryrun $RUN_ARGS
else
  # shellcheck disable=SC2086
  APP_ID="$(torchx run --wait $RUN_ARGS)"
  torchx status "$APP_ID"
  torchx describe "$APP_ID"
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
fi
