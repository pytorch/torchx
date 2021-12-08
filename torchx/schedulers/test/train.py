#!/usr/bin/env python3
# (c) Facebook, Inc. and its affiliates. Confidential and proprietary.

import argparse
import os
import sys
from typing import List

import torch
import torch.distributed as dist
import torch.nn.functional as F


def compute_world_size() -> int:

    rank = int(os.getenv("RANK"))
    world_size = int(os.getenv("WORLD_SIZE"))
    master_addr = os.getenv("MASTER_ADDR")
    master_port = int(os.getenv("MASTER_PORT"))
    backend = "gloo"

    print(f"initializing `{backend}` process group")
    dist.init_process_group(
        backend=backend,
        init_method=f"tcp://{master_addr}:{master_port}",
        rank=rank,
        world_size=world_size,
    )
    print("successfully initialized process group")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    t = F.one_hot(torch.tensor(rank), num_classes=world_size)
    dist.all_reduce(t)
    computed_world_size = int(torch.sum(t).item())
    print(
        f"rank: {rank}, actual world_size: {world_size}, computed world_size: {computed_world_size}"
    )
    return computed_world_size


def main() -> None:
    compute_world_size()


if __name__ == "__main__":
    main()
