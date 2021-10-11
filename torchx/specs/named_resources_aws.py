# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

r"""
`torchx.specs.named_resources_aws` contains resource definitions that represent corresponding AWS instance types
taken from https://aws.amazon.com/ec2/instance-types/. The resources are exposed
via entrypoints after installing torchx lib. The mapping is stored in the `setup.py` file.

The named resources currently do not specify AWS instance type capabilities but merely represent
the equvalent resource in mem, cpu and gpu numbers.

.. note::
    These resource definitions may change in future. It is expected for each user to
    manage their own resources. Follow https://pytorch.org/torchx/latest/specs.html#torchx.specs.get_named_resources
    to set up named resources.

Usage:

    ::

     from torchx.specs import named_resources
     print(named_resources["aws_t3.medium"])
     print(named_resources["aws_m5.2xlarge"])
     print(named_resources["aws_p3.2xlarge"])
     print(named_resources["aws_p3.8xlarge"])

"""

from torchx.specs.api import Resource

GiB: int = 1024


def aws_p3_2xlarge() -> Resource:
    return Resource(
        cpu=8,
        gpu=1,
        memMB=61 * GiB,
        capabilities={},
    )


def aws_p3_8xlarge() -> Resource:
    return Resource(
        cpu=32,
        gpu=4,
        memMB=244 * GiB,
        capabilities={},
    )


def aws_t3_medium() -> Resource:
    return Resource(
        cpu=2,
        gpu=0,
        memMB=4 * GiB,
        capabilities={},
    )


def aws_m5_2xlarge() -> Resource:
    return Resource(
        cpu=8,
        gpu=0,
        memMB=32 * GiB,
        capabilities={},
    )
