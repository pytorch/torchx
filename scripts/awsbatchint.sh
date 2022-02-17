#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

RUN_ARGS="--scheduler aws_batch -c queue=torchx,image_repo=495572122715.dkr.ecr.us-west-2.amazonaws.com/torchx/integration-tests utils.echo"

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
  LINES="$(torchx log "$APP_ID" | wc -l)"

  if [ "$LINES" -ne 1 ]
  then
      echo "expected 1 log lines"
      exit 1
  fi
fi
