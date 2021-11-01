# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchx.components.metrics as metrics
import torchx.specs as specs
from torchx.components.component_test_base import ComponentTestCase


class MetricsComponentTest(ComponentTestCase):
    def test_tensorboard(self) -> None:
        self.validate(metrics, "tensorboard")

        app = metrics.tensorboard(
            logdir="foo/logs",
            image="foo:bar",
            timeout=60.0,
            port=1234,
        )
        want = specs.AppDef(
            name="tensorboard",
            roles=[
                specs.Role(
                    name="tensorboard",
                    image="foo:bar",
                    entrypoint="python",
                    args=[
                        "-m",
                        "torchx.apps.utils.process_monitor",
                        "--timeout",
                        "60.0",
                        "--start_on_file",
                        "",
                        "--exit_on_file",
                        "",
                        "--",
                        "tensorboard",
                        "--bind_all",
                        "--port",
                        "1234",
                        "--logdir",
                        "foo/logs",
                    ],
                    port_map={"http": 1234},
                )
            ],
        )
        self.assertEqual(app, want)
