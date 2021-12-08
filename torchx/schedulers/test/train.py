#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional


import torch
import torch.distributed
import torch.nn.functional as F


def compute_world_size() -> int:

    rank = int(os.getenv("RANK")) # pyre-ignore[6]:
    world_size = int(os.getenv("WORLD_SIZE")) # pyre-ignore[6]:
    master_port = int(os.getenv("MASTER_PORT")) # pyre-ignore[6]:
    master_addr = os.getenv("MASTER_ADDR")
    backend = "gloo"

    print(f"initializing `{backend}` process group")
    torch.distributed.init_process_group(
        backend=backend,
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )
    print("successfully initialized process group")

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    t = F.one_hot(torch.tensor(rank), num_classes=world_size)
    torch.distributed.all_reduce(t)
    computed_world_size = int(torch.sum(t).item())
    print(
        f"rank: {rank}, actual world_size: {world_size}, computed world_size: {computed_world_size}"
    )
    return computed_world_size


def main() -> None:
    compute_world_size()


if __name__ == "__main__":
    main()
