# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
TorchX provides a standardize plugin interface to load dynamic dependencies at runtime.

This allows for extending standard app images with new dependencies via `pip
install` or otherwise that can then be dynamically loaded without requiring
code changes.

Configuration
-----------------

The init_plugins automatically loads a configuration file located at
`/etc/torchx/config.yaml` or from the path specified by `TORCHX_CONFIG`.

The config looks like this:

.. code:: yaml

    plugins:
      torchx.aws.s2: null
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

"""

import importlib
import os
from typing import Optional, Dict

import yaml

TORCHX_CONFIG_ENV: str = "TORCHX_CONFIG"
DEFAULT_TORCHX_CONFIG_PATH = "/etc/torchx/config.yaml"


def init_plugins(config_path: Optional[str] = None) -> None:
    """
    init_plugins loads the plugins from the specified config, the path
    specified by TORCHX_CONFIG environment variable or the default location
    at /etc/torchx/config.yaml.
    """
    if not config_path:
        config_path = os.getenv(TORCHX_CONFIG_ENV, DEFAULT_TORCHX_CONFIG_PATH)
    print(f"config path: {config_path}")

    if not os.path.exists(config_path):
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    init_plugins_from_config(config)


def init_plugins_from_config(config: Dict[str, object]) -> None:
    """
    init_plugins_from_config imports all of the plugins listed in provided config.
    """
    if plugins := config.get("plugins"):
        if not isinstance(plugins, dict):
            raise TypeError(f"plugins must be a dict: {plugins}")

        for provider, args in plugins.items():
            print(f"loading plugin: {provider}")
            module = importlib.import_module(provider)
            # pyre-fixme[16]: `ModuleType` has no attribute `init_plugin`.
            module.init_plugin(args)
