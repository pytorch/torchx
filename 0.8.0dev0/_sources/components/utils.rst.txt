Utils
================

.. automodule:: torchx.components.utils
.. currentmodule:: torchx.components.utils

.. autofunction:: echo
.. autofunction:: touch
.. autofunction:: sh
.. autofunction:: copy
.. autofunction:: python

Usage is very similar to just regular python, except that this supports remote launches. Example:

.. code-block:: bash

    # locally (cmd)
    $ torchx run utils.python --image $FBPKG -c "import torch; print(torch.__version__)"

    # locally (module)
    $ torchx run utils.python --image $FBPKG -m foo.bar.main

    # remote (cmd)
    $ torchx run -s mast utils.python --image $FBPKG -c "import torch; print(torch.__version__)"

    # remote (module)
    $ torchx run -s mast utils.python --image $FBPKG -m foo.bar.main

Notes:

* ``torchx run`` patches the current working directory (CWD) on top of ``$FBPKG`` for faster remote iteration.
* Patch contents will contain all changes to local fbcode however, the patch building is only triggered if CWD is a subdirectory of fbcode. If you are running from the root of fbcode (e.g. ~/fbsource/fbcode) your job will NOT be patched!
* Be careful not to abuse ``-c CMD``. Schedulers have a length limit on the arguments, hence don't try to pass long CMDs, use it sparingly.
* In `-m MODULE`, the module needs to be rooted off of fbcode. Example: for ~/fbsource/fbcode/foo/bar/main.py the module is ``-m foo.bar.main``.
* DO NOT override ``base_module`` in ``python_library`` buck rule. If you do, you are on your own, patching won't work.

Inline Script in Component

.. note::
    IMPORTANT: DO NOT ABUSE THIS FEATURE! This use be used sparringly and not abused! We reserve the right to remove this feature in the future.

A nice side effect of how TorchX and penv python is built is that you can do pretty much anything that you would normally do with python, with the added benefit that it auto patches your working directory and gives you the ability to run locally and remotely.
This means that python ``-c CMD`` will also work. Here's an example illustrating this

.. code-block:: console

    $ cd ~/fbsource/fbcode/torchx/examples/apps

    $ ls
    component.py  config  main.py  module  README.md  TARGETS

    # lets try getting the version of torch from a prebuilt fbpkg or bento kernel
    $ torchx run utils.python --image bento_kernel_pytorch_lightning -c "import torch; print(torch.__version__)"
    torchx 2021-10-27 11:27:28 INFO     loaded configs from /data/users/kiuk/fbsource/fbcode/torchx/fb/example/.torchxconfig
    2021-10-27 11:27:44,633 fbpkg.fetch INFO: completed download of bento_kernel_pytorch_lightning:405
    2021-10-27 11:27:44,634 fbpkg.fetch INFO: extracted bento_kernel_pytorch_lightning:405 to bento_kernel_pytorch_lightning
    2021-10-27 11:27:48,591 fbpkg.util WARNING: removing old version /home/kiuk/.torchx/fbpkg/bento_kernel_pytorch_lightning/403
    All packages downloaded successfully
    local_penv://torchx/torchx_utils_python_6effc4e2
    torchx 2021-10-27 11:27:49 INFO     Waiting for the app to finish...
    1.11.0a0+fb
    torchx 2021-10-27 11:27:58 INFO     Job finished: SUCCEEDED
    Now for a more interesting example, lets run a dumb all reduce of a 1-d tensor on 1 worker:
    $ torchx run utils.python --image torchx_fb_example \
    -c "import torch; import torch.distributed as dist; dist.init_process_group(backend='gloo', init_method='tcp://localhost:29500', rank=0, world_size=1); t=torch.tensor(1); dist.all_reduce(t); print(f'all reduce result: {t.item()}')"

    torchx 2021-10-27 10:23:05 INFO     loaded configs from /data/users/kiuk/fbsource/fbcode/torchx/fb/example/.torchxconfig
    2021-10-27 10:23:09,339 fbpkg.fetch INFO: checksums verified: torchx_fb_example:11
    All packages verified
    local_penv://torchx/torchx_utils_python_08a41456
    torchx 2021-10-27 10:23:09 INFO     Waiting for the app to finish...
    all reduce result: 1
    torchx 2021-10-27 10:23:13 INFO     Job finished: SUCCEEDED
    WARNING: Long inlined scripts won't work since schedulers typically have a character limit on the length of each argument.


.. autofunction:: booth
.. autofunction:: binary
