# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

r"""
`torchx.specs.named_resources_tpu` contains resource definitions that represent
corresponding Google Cloud TPU VMs.

TPUs require a matching torch version so the named resources will read the local
Torch version to set the `tf-version.cloud-tpus.google.com` annotation correctly.

.. note::
    These resource definitions may change in future. It is expected for each user to
    manage their own resources. Follow https://pytorch.org/torchx/latest/specs.html#torchx.specs.get_named_resources
    to set up named resources.

Usage:

.. doctest::

     from torchx.specs import named_resources
     print(named_resources["tpu_v2_8"])
     print(named_resources["tpu_v3_8"])
     print(named_resources["tpu_preemptible_v3_8"])
     print(named_resources["tpu_v3_2048"])
"""

from typing import Dict, Callable, Optional

from torchx.specs.api import Resource

NAMED_RESOURCES: Dict[str, Callable[[], Resource]] = {}


def _get_tf_version(version: Optional[str] = None) -> str:
    if version is None:
        try:
            from torch.version import __version__

            version = __version__
        except ImportError:
            version = "1.11"
    if "dev" in version:
        return "pytorch-nightly"
    short_ver = ".".join(version.split(".")[:2])
    return f"pytorch-{short_ver}"


def _register_type(ver: str, cores: int) -> Callable[[], Resource]:
    device: str = "cloud-tpus.google.com/" + ver

    def resource() -> Resource:
        return Resource(
            cpu=0,
            memMB=0,
            gpu=0,
            capabilities={
                "tf-version.cloud-tpus.google.com": _get_tf_version(),
            },
            devices={
                device: int(cores),
            },
        )

    resource_name = f"tpu_{ver.replace('-', '_')}_{cores}"
    NAMED_RESOURCES[resource_name] = resource
    return resource


tpu_v2_8: Callable[[], Resource] = _register_type("v2", 8)
tpu_preemptible_v2_8: Callable[[], Resource] = _register_type("preemptible-v2", 8)
tpu_v2_32: Callable[[], Resource] = _register_type("v2", 32)
tpu_v2_128: Callable[[], Resource] = _register_type("v2", 128)
tpu_v2_256: Callable[[], Resource] = _register_type("v2", 256)
tpu_v2_512: Callable[[], Resource] = _register_type("v2", 512)

tpu_v3_8: Callable[[], Resource] = _register_type("v3", 8)
tpu_preemptible_v3_8: Callable[[], Resource] = _register_type("preemptible-v3", 8)
tpu_v3_32: Callable[[], Resource] = _register_type("v3", 32)
tpu_v3_64: Callable[[], Resource] = _register_type("v3", 64)
tpu_v3_128: Callable[[], Resource] = _register_type("v3", 128)
tpu_v3_256: Callable[[], Resource] = _register_type("v3", 256)
tpu_v3_512: Callable[[], Resource] = _register_type("v3", 512)
tpu_v3_1024: Callable[[], Resource] = _register_type("v3", 1024)
tpu_v3_2048: Callable[[], Resource] = _register_type("v3", 2048)
