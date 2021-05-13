#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import unittest
from unittest.mock import patch

from torchelastic.tsm.driver.api import Application, Container, Resource, ElasticRole
from torchx.cli.cmd_describe import CmdDescribe


class CmdStatusTest(unittest.TestCase):
    def get_test_app(self) -> Application:
        resource = Resource(cpu=2, gpu=0, memMB=256)
        trainer = (
            ElasticRole("elastic_trainer", nnodes="2:3")
            .runs("trainer.par", "--arg1", "foo")
            .on(Container("trainer_fbpkg").require(resource))
            .replicas(2)
        )

        return Application("my_train_job").of(trainer)

    def test_run(self) -> None:
        parser = argparse.ArgumentParser()
        cmd_describe = CmdDescribe()
        cmd_describe.add_arguments(parser)
        args = parser.parse_args(["local://test_session/test_app"])

        for app in [None, self.get_test_app()]:
            with self.subTest(app=app):
                with patch(
                    "torchelastic.tsm.driver.standalone_session.StandaloneSession.describe",
                    return_value=app,
                ) as desc_mock:
                    cmd_describe.run(args)
                    desc_mock.assert_called_once_with(args.app_handle)
