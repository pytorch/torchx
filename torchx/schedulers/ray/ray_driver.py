# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import ray
from ray.train.utils import get_address_and_port
from ray.util.placement_group import PlacementGroup

if TYPE_CHECKING:
    from torchx.schedulers.ray.ray_common import RayActor, TORCHX_RANK0_HOST

# Hack to make code work for tests as well as running ray job.
# For tests the `torchx.schedulers.ray.ray_common` import must be used
# For running ray jobs `ray_common` import must be used
try:
    # pyre-fixme[21]: Could not find a module corresponding to import `ray_common`.
    from ray_common import RayActor, TORCHX_RANK0_HOST  # noqa: F811
except ModuleNotFoundError:
    from torchx.schedulers.ray.ray_common import RayActor, TORCHX_RANK0_HOST

_logger: logging.Logger = logging.getLogger(__name__)
_logger.setLevel(logging.getLevelName(os.environ.get("LOGLEVEL", "INFO")))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


@ray.remote
class CommandActor:  # pragma: no cover
    def __init__(self, command: List[str], env: Dict[str, str]) -> None:
        self.cmd: List[str] = command
        self.env = env
        self.master_addr: Optional[str] = None
        self.master_port: Optional[int] = None

    def exec_module(self) -> None:
        if self.master_addr is None or self.master_port is None:
            raise RuntimeError(
                "Either MASTER_ADDR or MASTER_PORT are not set. This is most likely bug in torchx"
                "Open issue at https://github.com/pytorch/torchx"
            )
        worker_evn = {}
        worker_evn.update(os.environ)
        worker_evn.update(self.env)
        worker_evn[TORCHX_RANK0_HOST] = self.master_addr
        popen = subprocess.Popen(self.cmd, env=worker_evn)

        returncode = popen.wait()
        _logger.info(f"Finished with code {returncode}")

        if returncode != 0:
            raise RuntimeError(f"exec_module failed with return code {returncode}")

    def get_actor_address_and_port(self) -> Tuple[str, int]:
        return get_address_and_port()

    def set_address_and_port(self, address: str, port: int) -> None:
        self.master_addr = address
        self.master_port = port


def load_actor_json(filename: str) -> List[RayActor]:
    with open(filename) as f:
        actors: List[RayActor] = []
        # Yes this is gross but it works
        actor_dict = json.load(f)
        actor_dict = json.loads(actor_dict)
        for actor in actor_dict:
            actors.append(RayActor(**actor))
        return actors


def create_placement_group(replicas: List[RayActor]) -> PlacementGroup:
    bundles = []
    for replica in replicas:
        bundles.append({"CPU": replica.num_cpus, "GPU": replica.num_gpus})

    # To change the strategy type
    # refer to available options here https://docs.ray.io/en/latest/placement-group.html#pgroup-strategy
    pg = ray.util.placement_group(bundles, strategy="SPREAD")

    _logger.info("Waiting for placement group to start.")
    ready = pg.wait(timeout_seconds=100)

    if not ready:  # pragma: no cover
        raise TimeoutError(
            "Placement group creation timed out. Make sure "
            "your cluster either has enough resources or use "
            "an autoscaling cluster. Current resources "
            "available: {}, resources requested by the "
            "placement group: {}".format(ray.available_resources(), pg.bundle_specs)
        )
    return pg


def create_command_actors(
    actors: List[RayActor], pg: PlacementGroup
) -> List[CommandActor]:
    cmd_actors: List[CommandActor] = []
    for i, replica in enumerate(actors):
        # Environment variables for distributed training
        actor = CommandActor.options(  # pyre-ignore[16]
            placement_group=pg,
            num_cpus=replica.num_cpus,
            num_gpus=replica.num_gpus,
        ).remote(replica.command, replica.env)
        cmd_actors.append(actor)

        if i == 0:
            rank_0_address = "localhost"
            rank_0_port = 0
        else:
            rank_0_address, rank_0_port = ray.get(
                # pyre-ignore[16]
                cmd_actors[0].get_actor_address_and_port.remote()
            )
        ray.get(actor.set_address_and_port.remote(rank_0_address, rank_0_port))

    return cmd_actors


def main() -> None:  # pragma: no cover
    actors: List[RayActor] = load_actor_json("actors.json")
    # pyre-fixme[16]: Module `worker` has no attribute `init`.
    ray.init(address="auto", namespace="torchx-ray")
    pg: PlacementGroup = create_placement_group(actors)
    command_actors: List[CommandActor] = create_command_actors(actors, pg)

    active_workers = [
        command_actor.exec_module.remote()  # pyre-ignore
        for command_actor in command_actors
    ]

    # Await return result of remote ray function
    while len(active_workers) > 0:
        _logger.info(f"running ray.wait on {active_workers}")

        # pyre-fixme[16]: Module `worker` has no attribute `wait`.
        completed_workers, active_workers = ray.wait(active_workers)
        # If a failure occurs the ObjectRef will be marked as completed.
        # Calling ray.get will expose the failure as a RayActorError.
        for object_ref in completed_workers:
            ray.get(object_ref)


if __name__ == "__main__":
    main()
