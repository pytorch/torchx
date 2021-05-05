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
`/etc/torchx.yaml` or from the path specified by `TORCHX_CONFIG`.

The config looks like this:

.. code:: yaml

    storage_providers:
      - torchx.aws.s3

Configuration options:

- storage_providers: this is a list of python packages that should be loaded at runtime to register any third party storage_providers.


Extending
-----------------

You can extend the prebuilt docker container to add extra dependencies,
components or storage providers.

.. code:: Dockerfile
    FROM pytorch/torchx:latest

    RUN pip install <your package>
    COPY torchx.yaml /etc/torchx.yaml

This container can then be used instead of the default by specifying the
`TORCHX_CONTAINER` environment variable with the kfp adapter.
"""
