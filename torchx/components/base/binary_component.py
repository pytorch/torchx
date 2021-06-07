# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

from torchx.specs import api


def binary_component(
    name: str,
    image: str,
    entrypoint: str,
    args: Optional[List[str]] = None,
    env: Optional[Dict[str, str]] = None,
) -> api.AppDef:
    """
    binary_component creates a single binary and container component from the
    provided arguments.

    >>> from torchx.components.base.binary_component import binary_component
    >>> binary_component(
    ...     name="datapreproc",
    ...     image="pytorch/pytorch:latest",
    ...     entrypoint="python3",
    ...     args=["--version"],
    ...     env={
    ...         "FOO": "bar",
    ...     },
    ... )
    AppDef(name='datapreproc', ...)
    """

    return api.AppDef(
        name=name,
        roles=[
            api.Role(
                name=name,
                entrypoint=entrypoint,
                args=args or [],
                env=env or {},
                container=api.Container(
                    image=image,
                ),
            ),
        ],
    )
