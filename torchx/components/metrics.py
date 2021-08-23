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
