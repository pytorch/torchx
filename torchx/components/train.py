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

Check out the code for :ref:`examples_apps/lightning_classy_vision/train:Trainer App Example`.
You can try it out by running a single trainer example on your desktop:

.. note:

    Follow :ref:`examples_apps/lightning_classy_vision/component:Prerequisites of running examples` to
    before running the examples


.. code:: bash

    cd ~/torchx
    python torchx/examples/apps/lightning_classy_vision/train.py --skip_export


Torchx simplifies application execution by providing a simple to use APIs that standardize
application execution on local or remote environments. It does this by introducing a concept of a
Component.

Each user application should be accompanied with the corresponding component.
Check out the single node trainer code:
:ref:`examples_apps/lightning_classy_vision/component:Trainer Component`

Try it out yourself:

.. code:: bash

    cd torchx

    torchx run -s local_cwd \
./torchx/examples/apps/lightning_classy_vision/component.py:trainer --skip-export


The code above will execute a single trainer on a user desktop.
If you have docker installed on your laptop you can running the same single trainer via the following cmd:

.. code:: bash

    cd torchx

    torchx run -s local_docker \
./torchx/examples/apps/lightning_classy_vision/component.py:trainer --skip-export


You can learn more about authoring your own components.

Torchx has great support for simplifying execution of distributed jobs, that you can learn more here.

"""
