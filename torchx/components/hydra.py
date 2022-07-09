# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import os.path

import torchx.specs as specs
from hydra import compose, initialize_config_dir
from torchx.specs.finder import get_component
from omegaconf import OmegaConf


def run(
    config_path: str,
    config_name: str,
    *overrides: str,
) -> specs.AppDef:
    """
    run executes a component using a hydra loaded arguments.

    This replaces .torchxconfig
    """
    initialize_config_dir(config_dir=os.path.join(os.getcwd(), config_path))
    cfg = compose(config_name=config_name, overrides=overrides)

    torchx_cfg = cfg["torchx"]
    comp = get_component(torchx_cfg["component"]).fn

    kwargs = dict(torchx_cfg["args"].items())
    args = list(overrides)
    if "script_args" in kwargs:
        args = kwargs["script_args"] + args
        del kwargs["script_args"]

    return comp(*args, **kwargs)

def app(
    config_path: str,
    config_name: str,
    *overrides: str,
) -> specs.AppDef:
    """
    app runs a appdef specified in the hydra config.

    This replaces .torchxconfig and components.
    """
    initialize_config_dir(config_dir=os.path.join(os.getcwd(), config_path))
    cfg = compose(config_name=config_name, overrides=overrides)

    torchx_cfg = cfg["torchx"]
    app_cfg = torchx_cfg["app"]

    from dacite import from_dict

    return from_dict(data_class=specs.AppDef, data=OmegaConf.to_object(app_cfg))

