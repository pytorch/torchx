# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
torchx provides a standard container spec and entry point at
`torchx/container/main.py`. This allows for executing torchx components by
fully qualified class name.

Usage
-----------------

The container entry point lives at `torchx/container/main.py`.

The first argument is the fully qualified class name. The entry point will automatically import that path and load the component config, inputs and output definitions from the command line args.

Ex:

.. code:: bash

    $ docker run -it --name torchx --rm pytorch/torchx:latest python3 torchx/container/main.py torchx.components.io.copy.Copy --input_path 'file:///etc/os-release' --output_path 'file:///tmp/bar'


Configuration
-----------------

The entry point automatically loads a configuration file located at
`/etc/torchx/config.yaml` or from the path specified by `TORCHX_CONFIG`.

The config looks like this:

.. code:: yaml

    plugins:
      torchx.aws.s3: null
      your_plugin:
        foo: bar


Configuration options:

**plugins**

This is a list of python packages that should be loaded at runtime to register any third party plugins.
The init_plugin method of the module will be called with the parsed yaml options from the plugin config.

.. code:: python

    from torchx.sdk.storage import register_storage_provider

    def init_plugin(args):
        register_storage_provider(<your provider>)


Extending
-----------------

You can extend the prebuilt docker container to add extra dependencies,
components or storage providers.

.. code:: Dockerfile
    FROM pytorch/torchx:latest

    RUN pip install <your package>
    COPY torchx.yaml /etc/torchx/config.yaml

This container can then be used instead of the default by specifying the
`TORCHX_CONTAINER` environment variable with the kfp adapter.
"""
