# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
For metrics we recommend using Tensorboard to log metrics directly to cloud
storage along side your model. As the model trains you can use launch a
tensorboard instance locally to monitor your model progress:

.. code-block:: shell-session

 $ tensorboard --log-dir provider://path/to/logs

See the :ref:`examples_apps/lightning_classy_vision/train:Trainer App Example` for an example on how to use the PyTorch
Lightning TensorboardLogger.

A TorchX Tensorboard builtin component is being tracked via
https://github.com/pytorch/torchx/issues/128.

Reference
----------------

* PyTorch Tensorboard Tutorial https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html
* PyTorch Lightning Loggers https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html

"""

import torchx
import torchx.specs as specs


def tensorboard(
    logdir: str,
    image: str = torchx.IMAGE,
    timeout: float = 60 * 60,  # 1 hour
    port: int = 6006,
    start_on_file: str = "",
    exit_on_file: str = "",
) -> specs.AppDef:
    """
    Runs a tensorboard server.

    Args:
        logdir: fsspec path to the Tensorboard logs
        image: image to use
        timeout: maximum time to run before exiting (seconds)
        start_on_file: start the server when the fsspec path is created
        exit_on_file: shutdown the server when the fsspec path is created
    """

    return specs.AppDef(
        name="tensorboard",
        roles=[
            specs.Role(
                name="tensorboard",
                image=image,
                entrypoint="python",
                args=[
                    "-m",
                    "torchx.apps.utils.process_monitor",
                    "--timeout",
                    str(timeout),
                    "--start_on_file",
                    start_on_file,
                    "--exit_on_file",
                    exit_on_file,
                    "--",
                    "tensorboard",
                    "--bind_all",
                    "--port",
                    str(port),
                    "--logdir",
                    logdir,
                ],
                port_map={"http": port},
            )
        ],
    )
