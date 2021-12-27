#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -ex

rm -r dist || true


python setup.py --override-name torchx-nightly bdist_wheel

if [ -z "$PYPI_TOKEN" ]; then
    echo "must specify PYPI_TOKEN"
    exit 1
fi

python3 -m twine upload \
    --username __token__ \
    --password "$PYPI_TOKEN" \
    dist/torchx_nightly-*
