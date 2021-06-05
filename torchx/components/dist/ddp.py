# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torchx.specs as specs


def get_app_spec(fixme: str) -> specs.AppDef:
    # TODO actually return ddp app (to get the resources right needs to be
    #  done after properly implementing torchx.components.base.named_resource)
    role = specs.ElasticRole(name="foobar")
    return specs.AppDef(name=fixme, roles=[role])
