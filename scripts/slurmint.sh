#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

DIST=dist

rm -r $DIST || true
python setup.py bdist_wheel

WHEEL="$DIST/$(ls $DIST)"

if [[ -z "${SLURM_INSTANCE_MASTER}" ]]; then
    echo "SLURM_INSTANCE_MASTER is not set, skipping test..."
    exit 0
fi

JOB="$USER-$(uuidgen)"
DIR="/tmp/$JOB"
VENV="$DIR/venv"

function run_cmd {
    # shellcheck disable=SC2048,SC2086
    mssh -o ServerAliveInterval=60 "$SLURM_INSTANCE_MASTER" -- $*
}

function run_scp {
    rsync -rav -e mssh "$1" "$SLURM_INSTANCE_MASTER:$2"
}

function cleanup {
  echo "Removing $DIR"
  run_cmd rm  -r "$DIR"
}
trap cleanup EXIT


REMOTE_WHEEL="$DIR/$(basename "$WHEEL")"

SCRIPT="scripts/slurmtest.sh"
REMOTE_SCRIPT="$DIR/$(basename "$SCRIPT")"

run_cmd mkdir "$DIR"
run_cmd virtualenv -p /usr/bin/python3.8 "$VENV"
run_scp "$WHEEL" "$REMOTE_WHEEL"
run_scp "$SCRIPT" "$REMOTE_SCRIPT"
run_cmd "$REMOTE_SCRIPT" "$REMOTE_WHEEL" "$VENV"
