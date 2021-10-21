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


1. :ref:`examples_apps/lightning_classy_vision/train:Trainer App Example`
2. :ref:`examples_apps/lightning_classy_vision/component:Trainer Component`
3. :ref:`component_best_practices:Component Best Practices`
4. See :py:mod:`torchx.components` to learn more about authoring components
5. For more information on distributed training see :py:mod:`torchx.components.dist`.

"""
