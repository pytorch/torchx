# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchx.specs as specs


def get_app_spec(msg: str = "hello world") -> specs.AppDef:
    """
    Echos a message to stdout (calls /bin/echo)

    Args:
        msg: message to echo

    """
    return specs.AppDef(
        name="echo",
        roles=[
            specs.Role(
                name="echo",
                image="/tmp",
                entrypoint="/bin/echo",
                args=[msg],
                num_replicas=1,
            )
        ],
    )
