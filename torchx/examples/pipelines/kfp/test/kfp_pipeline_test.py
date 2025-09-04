# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import os
import os.path
import sys
import tempfile
import unittest


class KFPPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dir = tempfile.TemporaryDirectory()  # noqa: P201
        self.orig_dir = os.getcwd()
        os.chdir(self.dir.name)

    def tearDown(self) -> None:
        os.chdir(self.orig_dir)
        self.dir.cleanup()

    def test_kfp_pipeline(self) -> None:
        sys.argv = [
            "advanced_pipeline.py",
            "--output_path",
            "bar",
        ]
        from torchx.examples.pipelines.kfp import advanced_pipeline  # noqa: F401

        self.assertTrue(os.path.exists("pipeline.yaml"))

    def test_intro_pipeline(self) -> None:
        sys.argv = ["intro_pipeline.py"]
        from torchx.examples.pipelines.kfp import intro_pipeline  # noqa: F401

        self.assertTrue(os.path.exists("pipeline.yaml"))

    def test_dist_pipeline(self) -> None:
        sys.argv = ["dist_pipeline.py"]
        from torchx.examples.pipelines.kfp import dist_pipeline  # noqa: F401

        self.assertTrue(os.path.exists("pipeline.yaml"))

    def test_task_configs_pipeline(self) -> None:
        sys.argv = ["task_configs_pipeline.py"]
        from torchx.examples.pipelines.kfp import task_configs_pipeline  # noqa: F401

        self.assertTrue(os.path.exists("task_configs_pipeline.yaml"))

    def test_dist_pipeline_v2_enhanced(self) -> None:
        sys.argv = ["dist_pipeline_v2_enhanced.py"]
        from torchx.examples.pipelines.kfp import (  # noqa: F401
            dist_pipeline_v2_enhanced,
        )

        self.assertTrue(os.path.exists("enhanced_distributed_pipeline.yaml"))
