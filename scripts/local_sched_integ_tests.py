#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Local scheduler integration tests.
"""

import os
import tempfile

import examples.apps as apps
from examples.apps.lightning_classy_vision.component import trainer
from torchx.components.component_test_base import ComponentTestCase
from torchx.components.utils import echo


class LocalSchedTest(ComponentTestCase):
    def _get_local_image(self):
        return os.path.dirname(apps.__file__)

    def test_lightning_classy_vision(self):
        image = self._get_local_image()
        with tempfile.TemporaryDirectory() as tmpdir:
            component_kwargs = {
                "image": image,
                "output_path": tmpdir,
                "use_test_data": True,
                "skip_export": True,
                "log_path": tmpdir,
            }
            self.run_component_on_local(trainer, component_kwargs=component_kwargs)

    def test_utils_echo(self):
        image = self._get_local_image()
        component_kwargs = {
            "msg": "test run",
            "image": image,
        }
        self.run_component_on_local(echo, component_kwargs=component_kwargs)


def main() -> None:
    test_suite = LocalSchedTest()
    test_suite.test_lightning_classy_vision()
    test_suite.test_utils_echo()


if __name__ == "__main__":
    main()
