# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
These components aim to make it easier to interact with inference and serving
tools such as `torchserve <https://pytorch.org/serve/>`_.
"""

from typing import Dict, Optional

import torchx.specs as specs
from torchx.version import TORCHX_IMAGE


def torchserve(
        model_path: str,
        management_api: str,
        image: str = TORCHX_IMAGE,
        params: Optional[Dict[str, object]] = None,
        dryrun: bool = False,
) -> specs.AppDef:
    """Deploys the provided model to the given torchserve management API
    endpoint.

    >>> from torchx.components.serve import torchserve
    >>> torchserve(
    ...     model_path="s3://your-bucket/your-model.pt",
    ...     management_api="http://torchserve:8081",
    ... )
    AppDef(name='torchx-serve-torchserve', ...)

    Args:
        model_path: The fsspec path to the model archive file.
        management_api: The URL to the root of the torchserve management API.
        image: Container to use.
        params: torchserve parameters.
            See https://pytorch.org/serve/management_api.html#register-a-model
        dryrun: Start the app, but does not perform actual work

    Returns:
        specs.AppDef: the Torchx application definition
    """

    args = [
        "-m",
        "torchx.apps.serve.serve",
        "--model_path",
        model_path,
        "--management_api",
        management_api,
    ]
    if dryrun:
        args.append("--dryrun")
    if params is not None:
        for param, value in params.items():
            args += [
                f"--{param}",
                str(value),
            ]

    return specs.AppDef(
        name="torchx-serve",
        roles=[
            specs.Role(
                name="torchx-serve-torchserve",
                image=image,
                entrypoint="python",
                args=args,
                port_map={"model-download": 8222},
            ),
        ],
    )
