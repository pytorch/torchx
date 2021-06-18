# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path
import sys
import tempfile
import unittest


class KFPPipelineTest(unittest.TestCase):
    def test_kfp_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_dir = os.getcwd()
            os.chdir(tmpdir)

            sys.argv = [
                "advanced_pipeline.py",
                "--data_path",
                "foo",
                "--output_path",
                "bar",
            ]
            from examples.pipelines.kfp import advanced_pipeline  # noqa: F401

            self.assertTrue(os.path.exists("pipeline.yaml"))

            os.chdir(orig_dir)

    def test_intro_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            orig_dir = os.getcwd()
            os.chdir(tmpdir)

            sys.argv = ["intro_pipeline.py"]
            from examples.pipelines.kfp import intro_pipeline  # noqa: F401

            self.assertTrue(os.path.exists("pipeline.yaml"))

            os.chdir(orig_dir)
