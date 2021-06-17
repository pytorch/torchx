#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchx.apps.io.copy import Copy
from torchx.runtime.storage import upload_blob, download_blob, temppath


class CopyTest(unittest.TestCase):
    def test_copy(self) -> None:
        with temppath() as input_path, temppath() as output_path:
            data = b"banana"
            upload_blob(input_path, data)
            c = Copy(input_path=input_path, output_path=output_path, a=10)
            c.run(c.inputs, c.outputs)
            out = download_blob(output_path)
            self.assertEqual(out, data)
