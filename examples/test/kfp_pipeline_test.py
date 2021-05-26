# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path
import tempfile
import unittest

from examples import kfp_pipeline


class ExamplesTest(unittest.TestCase):
    def test_kfp_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_dir = os.getcwd()
            os.chdir(tmpdir)

            kfp_pipeline.main(["--data_path", "foo", "--output_path", "bar"])
            self.assertTrue(os.path.exists("pipeline.yaml"))

            os.chdir(orig_dir)
