# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Training machine learning models often requires custom train loop and custom
code. As such, we don't provide an out of the box training loop app. We do
however have examples for how you can construct your training app as well as
generic components you can use to run your custom training app.


Trainer Examples
----------------

* :ref:`examples_apps/lightning_classy_vision/component:Single Trainer Component`
* :ref:`examples_apps/lightning_classy_vision/component:Distributed Trainer Component`
* :ref:`examples_apps/lightning_classy_vision/train:Trainer App Example`

Components
-----------

These are generic components for common patterns that that can be used in your
train components.

* Distributed Data Parallel Component - :meth:`torchx.components.dist.ddp`


"""
