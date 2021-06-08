# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Runs the example lightning_classy_vision app.
"""

import torchx.specs.api as torchx


def classy_vision(
    image: str,
    output_path: str,
    load_path: str = "",
    log_dir: str = "/logs",
) -> torchx.AppDef:
    """Runs the example lightning_classy_vision app.

    Runs the example lightning_classy_vision app.

    Args:
        image: image to run (e.g. foobar:latest)
        resource: resource spec
        output_path: output path for model checkpoints (e.g. file:///foo/bar)
        load_path: path to load pretrained model from
        log_dir: path to save tensorboard logs to
    """
    container = torchx.Container(image=image).require(
        torchx.Resource(cpu=1, gpu=1, memMB=1024)
    )
    entrypoint = "main"

    trainer_role = (
        torchx.Role(name="trainer")
        .runs(
            "main",
            "--output_path",
            output_path,
            "--load_path",
            load_path,
            "--log_dir",
            log_dir,
        )
        .on(container)
        .replicas(1)
    )

    return torchx.AppDef("examples-lightning_classy_vision").of(trainer_role)
