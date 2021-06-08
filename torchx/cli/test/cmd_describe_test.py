#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import unittest
from unittest.mock import patch

from torchx.cli.cmd_describe import CmdDescribe
from torchx.components.base import torch_dist_role
from torchx.specs.api import AppDef, Container, Resource


class CmdDescribeTest(unittest.TestCase):
    def get_test_app(self) -> AppDef:
        resource = Resource(cpu=2, gpu=0, memMB=256)
        trainer = torch_dist_role(
            "elastic_trainer",
            entrypoint="trainer.par",
            script_args=["--arg1", "foo"],
            num_replicas=2,
            container=Container("trainer_fbpkg").require(resource),
            nnodes="2:3",
        )
        return AppDef("my_train_job").of(trainer)

    def test_run(self) -> None:
        parser = argparse.ArgumentParser()
        cmd_describe = CmdDescribe()
        cmd_describe.add_arguments(parser)
        args = parser.parse_args(["local://test_session/test_app"])

        for app in [None, self.get_test_app()]:
            with self.subTest(app=app):
                with patch(
                    "torchx.runner.api.Runner.describe",
                    return_value=app,
                ) as desc_mock:
                    cmd_describe.run(args)
                    desc_mock.assert_called_once_with(args.app_handle)
