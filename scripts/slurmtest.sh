#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

REMOTE_WHEEL="$1"
VENV="$2"

# shellcheck disable=SC1091
source /opt/slurm/etc/slurm.sh
sbatch --version
# shellcheck disable=SC1090
source "$VENV"/bin/activate
python --version
pip install "$REMOTE_WHEEL"

APP_ID="$(torchx run --wait --scheduler slurm utils.echo --num_replicas 3)"
torchx status "$APP_ID"
torchx describe "$APP_ID"
LOG_FILE="slurm-$(basename "$APP_ID").out"
cat "$LOG_FILE"
LINES="$(wc -l "$LOG_FILE" | cut -d' ' -f1)"

if [ "$LINES" -ne 3 ]
then
    echo "expected 3 log lines"
    exit 1
fi
