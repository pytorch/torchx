# Copyright (c) Meta Platforms, Inc. and affiliates.
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

import warnings
from typing import Callable, Mapping

from torchx.specs.api import Resource

EFA_DEVICE = "vpc.amazonaws.com/efa"

# ecs and ec2 have memtax and currently AWS Batch uses hard memory limits
# so we have to account for mem tax when registering these resources for AWS
# otherwise the job will be stuck in the jobqueue forever
# 97% is based on empirical observation that works well for most instance types
# see: https://docs.aws.amazon.com/batch/latest/userguide/memory-management.html
MEM_TAX = 0.97

# determines instance type for non-honogeneous CEs
# see https://github.com/pytorch/torchx/issues/780
K8S_ITYPE = "node.kubernetes.io/instance-type"
GiB: int = int(1024 * MEM_TAX)


def instance_type_from_resource(resource: Resource) -> str:
    instance_type = resource.capabilities.get(K8S_ITYPE)
    if instance_type is None:
        warnings.warn(
            "Cannot determine resource instance type which can cause issues for non-homogeneous CEs and multinode jobs. Consider providing torchx.specs.named_resources_aws:K8S_TYPE resource capability."
        )
    return instance_type


def aws_p3_2xlarge() -> Resource:
    return Resource(
        cpu=8, gpu=1, memMB=61 * GiB, capabilities={K8S_ITYPE: "p3.2xlarge"}
    )


def aws_p3_8xlarge() -> Resource:
    return Resource(
        cpu=32, gpu=4, memMB=244 * GiB, capabilities={K8S_ITYPE: "p3.8xlarge"}
    )


def aws_p3_16xlarge() -> Resource:
    return Resource(
        cpu=64, gpu=8, memMB=488 * GiB, capabilities={K8S_ITYPE: "p3.16xlarge"}
    )


def aws_p3dn_24xlarge() -> Resource:
    return Resource(
        cpu=96,
        gpu=8,
        memMB=768 * GiB,
        capabilities={K8S_ITYPE: "p3dn.24xlarge"},
        devices={EFA_DEVICE: 1},
    )


def aws_p4d_24xlarge() -> Resource:
    return Resource(
        cpu=96,
        gpu=8,
        memMB=1152 * GiB,
        capabilities={K8S_ITYPE: "p4d.24xlarge"},
        devices={EFA_DEVICE: 4},
    )


def aws_p4de_24xlarge() -> Resource:
    # p4de has same cpu, gpu, memMB as p4d but gpu memory is 2x (32GB vs 64GB per GPU)
    return Resource(
        cpu=96,
        gpu=8,
        memMB=1152 * GiB,
        capabilities={K8S_ITYPE: "p4de.24xlarge"},
        devices={EFA_DEVICE: 4},
    )


def aws_t3_medium() -> Resource:
    return Resource(cpu=2, gpu=0, memMB=4 * GiB, capabilities={K8S_ITYPE: "t3.medium"})


def aws_m5_2xlarge() -> Resource:
    return Resource(
        cpu=8, gpu=0, memMB=32 * GiB, capabilities={K8S_ITYPE: "m5.2xlarge"}
    )


def aws_g4dn_xlarge() -> Resource:
    return Resource(
        cpu=4, gpu=1, memMB=16 * GiB, capabilities={K8S_ITYPE: "g4dn.xlarge"}
    )


def aws_g4dn_2xlarge() -> Resource:
    return Resource(
        cpu=8, gpu=1, memMB=32 * GiB, capabilities={K8S_ITYPE: "g4dn.2xlarge"}
    )


def aws_g4dn_4xlarge() -> Resource:
    return Resource(
        cpu=16, gpu=1, memMB=64 * GiB, capabilities={K8S_ITYPE: "g4dn.4xlarge"}
    )


def aws_g4dn_8xlarge() -> Resource:
    return Resource(
        cpu=32,
        gpu=1,
        memMB=128 * GiB,
        capabilities={K8S_ITYPE: "g4dn.8xlarge"},
        devices={EFA_DEVICE: 1},
    )


def aws_g4dn_12xlarge() -> Resource:
    return Resource(
        cpu=48,
        gpu=4,
        memMB=192 * GiB,
        capabilities={K8S_ITYPE: "g4dn.12xlarge"},
        devices={EFA_DEVICE: 1},
    )


def aws_g4dn_16xlarge() -> Resource:
    return Resource(
        cpu=64,
        gpu=1,
        memMB=256 * GiB,
        capabilities={K8S_ITYPE: "g4dn.16xlarge"},
        devices={EFA_DEVICE: 1},
    )


def aws_g4dn_metal() -> Resource:
    return Resource(
        cpu=96,
        gpu=8,
        memMB=384 * GiB,
        capabilities={K8S_ITYPE: "g4dn.metal"},
        devices={EFA_DEVICE: 1},
    )


def aws_g5_xlarge() -> Resource:
    return Resource(cpu=4, gpu=1, memMB=16 * GiB, capabilities={K8S_ITYPE: "g5.xlarge"})


def aws_g5_2xlarge() -> Resource:
    return Resource(
        cpu=8, gpu=1, memMB=32 * GiB, capabilities={K8S_ITYPE: "g5.2xlarge"}
    )


def aws_g5_4xlarge() -> Resource:
    return Resource(
        cpu=16, gpu=1, memMB=64 * GiB, capabilities={K8S_ITYPE: "g5.4xlarge"}
    )


def aws_g5_8xlarge() -> Resource:
    return Resource(
        cpu=32,
        gpu=1,
        memMB=128 * GiB,
        capabilities={K8S_ITYPE: "g5.8xlarge"},
        devices={EFA_DEVICE: 1},
    )


def aws_g5_12xlarge() -> Resource:
    return Resource(
        cpu=48,
        gpu=4,
        memMB=192 * GiB,
        capabilities={K8S_ITYPE: "g5.12xlarge"},
        devices={EFA_DEVICE: 1},
    )


def aws_g5_16xlarge() -> Resource:
    return Resource(
        cpu=64,
        gpu=1,
        memMB=256 * GiB,
        capabilities={K8S_ITYPE: "g5.16xlarge"},
        devices={EFA_DEVICE: 1},
    )


def aws_g5_24xlarge() -> Resource:
    return Resource(
        cpu=96,
        gpu=4,
        memMB=384 * GiB,
        capabilities={K8S_ITYPE: "g5.24xlarge"},
        devices={EFA_DEVICE: 1},
    )


def aws_g5_48xlarge() -> Resource:
    return Resource(
        cpu=192,
        gpu=8,
        memMB=768 * GiB,
        capabilities={K8S_ITYPE: "g5.48xlarge"},
        devices={EFA_DEVICE: 1},
    )


def aws_trn1_2xlarge() -> Resource:
    return Resource(
        cpu=8, gpu=0, memMB=32 * GiB, capabilities={K8S_ITYPE: "trn1.2xlarge"}
    )


def aws_trn1_32xlarge() -> Resource:
    return Resource(
        cpu=128,
        gpu=0,
        memMB=512 * GiB,
        capabilities={K8S_ITYPE: "trn1.32xlarge"},
        devices={EFA_DEVICE: 8},
    )


NAMED_RESOURCES: Mapping[str, Callable[[], Resource]] = {
    "aws_t3.medium": aws_t3_medium,
    "aws_m5.2xlarge": aws_m5_2xlarge,
    "aws_p3.2xlarge": aws_p3_2xlarge,
    "aws_p3.8xlarge": aws_p3_8xlarge,
    "aws_p3.16xlarge": aws_p3_16xlarge,
    "aws_p3dn.24xlarge": aws_p3dn_24xlarge,
    "aws_p4d.24xlarge": aws_p4d_24xlarge,
    "aws_p4de.24xlarge": aws_p4de_24xlarge,
    "aws_g4dn.xlarge": aws_g4dn_xlarge,
    "aws_g4dn.2xlarge": aws_g4dn_2xlarge,
    "aws_g4dn.4xlarge": aws_g4dn_4xlarge,
    "aws_g4dn.8xlarge": aws_g4dn_8xlarge,
    "aws_g4dn.16xlarge": aws_g4dn_16xlarge,
    "aws_g4dn.12xlarge": aws_g4dn_12xlarge,
    "aws_g4dn.metal": aws_g4dn_metal,
    "aws_g5.xlarge": aws_g5_xlarge,
    "aws_g5.2xlarge": aws_g5_2xlarge,
    "aws_g5.4xlarge": aws_g5_4xlarge,
    "aws_g5.8xlarge": aws_g5_8xlarge,
    "aws_g5.16xlarge": aws_g5_16xlarge,
    "aws_g5.12xlarge": aws_g5_12xlarge,
    "aws_g5.24xlarge": aws_g5_24xlarge,
    "aws_g5.48xlarge": aws_g5_48xlarge,
    "aws_trn1.2xlarge": aws_trn1_2xlarge,
    "aws_trn1.32xlarge": aws_trn1_32xlarge,
}
