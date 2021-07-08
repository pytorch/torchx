# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Utility components that are `ready-to-use` out of the box. These are
components that simply execute well known binaries (e.g. ``/bin/cp``)
and are meant to be used as tutorial materials or glue operations between
meaningful stages in a workflow.
"""

import torchx.specs as specs


def echo(
    msg: str = "hello world", image: str = "/tmp", num_replicas: int = 1
) -> specs.AppDef:
    """
    Echos a message to stdout (calls /bin/echo)

    Args:
        msg: message to echo
        image: image to use
        num_replicas: number of replicas to run

    """
    return specs.AppDef(
        name="echo",
        roles=[
            specs.Role(
                name="echo",
                image=image,
                entrypoint="/bin/echo",
                args=[msg],
                num_replicas=num_replicas,
            )
        ],
    )


def touch(file: str) -> specs.AppDef:
    """
    Touches a file (calls /bin/touch)

    Args:
        file: file to create

    """
    return specs.AppDef(
        name="touch",
        roles=[
            specs.Role(
                name="touch",
                image="/tmp",
                entrypoint="/bin/touch",
                args=[file],
                num_replicas=1,
            )
        ],
    )
