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
from typing import Dict, List, Tuple, Optional

import ray
from ray.train.utils import get_address_and_port
from ray.util.placement_group import PlacementGroup

# Hack to make code work for tests as well as running ray job.
# For tests the `torchx.schedulers.ray.ray_common` import must be used
# For running ray jobs `ray_common` import must be used
try:
    # pyre-ignore[21]: Could not find a module corresponding to import `ray_common`
    from ray_common import RayActor
except ModuleNotFoundError:
    from torchx.schedulers.ray.ray_common import RayActor

_logger: logging.Logger = logging.getLogger(__name__)
_logger.setLevel(logging.getLevelName(os.environ.get("LOGLEVEL", "INFO")))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


@ray.remote
class CommandActor:  # pragma: no cover
    def __init__(self, command: str, env: Dict[str, str]) -> None:
        self.cmd: List[str] = command.split(" ")
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
        worker_evn["MASTER_ADDR"] = self.master_addr
        worker_evn["MASTER_PORT"] = str(self.master_port)
        popen = subprocess.Popen(self.cmd, env=worker_evn)
        returncode = popen.wait()
        _logger.info(f"Finished with code {returncode}")

    def get_actor_address_and_port(self) -> Tuple[str, int]:
        return get_address_and_port()

    def set_address_and_port(self, address: str, port: int) -> None:
        self.master_addr = address
        self.master_port = port


# pyre-ignore[11]
def load_actor_json(filename: str) -> List["RayActor"]:
    with open(filename) as f:
        actors: List[RayActor] = []
        # Yes this is gross but it works
        actor_dict = json.load(f)
        actor_dict = json.loads(actor_dict)
        for actor in actor_dict:
            actors.append(RayActor(**actor))
        return actors


def create_placement_groups(actors: List[RayActor]) -> List[PlacementGroup]:
    pgs: List[PlacementGroup] = []
    for actor in actors:
        bundle = {"CPU": actor.num_cpus, "GPU": actor.num_gpus}
        bundles = [bundle] * actor.num_replicas

        # To change the strategy type
        # refer to available options here https://docs.ray.io/en/latest/placement-group.html#pgroup-strategy
        pg = ray.util.placement_group(bundles, strategy="SPREAD")
        pgs.append(pg)

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
    return pgs


def create_command_actors(
    actors: List[RayActor], pgs: List[PlacementGroup]
) -> List[CommandActor]:
    cmd_actors: List[CommandActor] = []
    for i in range(len(actors)):
        world_size = actors[i].num_replicas
        actor_group = []

        for rank in range(world_size):

            # Environment variables for distributed training
            rank_env = {
                "WORLD_SIZE": str(world_size),
                "RANK": str(rank),
            }

            actor_and_rank_env = {**actors[i].env, **rank_env}

            actor_group.append(
                CommandActor.options(  # pyre-ignore[16]
                    placement_group=pgs[i],
                    num_cpus=actors[i].num_cpus,
                    num_gpus=actors[i].num_gpus,
                ).remote(actors[i].command, actor_and_rank_env)
            )

            rank_0_address, rank_0_port = ray.get(
                actor_group[0].get_actor_address_and_port.remote()
            )

            for actor in actor_group:
                ray.get(actor.set_address_and_port.remote(rank_0_address, rank_0_port))

            cmd_actors.extend(actor_group)

    return cmd_actors


def main() -> None:  # pragma: no cover
    actors: List[RayActor] = load_actor_json("actors.json")
    # pyre-fixme[16]: Module `worker` has no attribute `init`.
    ray.init(address="auto", namespace="torchx-ray")
    pgs: List[PlacementGroup] = create_placement_groups(actors)
    command_actors: List[CommandActor] = create_command_actors(actors, pgs)

    active_workers = [
        command_actor.exec_module.remote()  # pyre-ignore
        for command_actor in command_actors
    ]

    # Await return result of remote ray function
    while len(active_workers) > 0:
        # pyre-fixme[16]: Module `worker` has no attribute `wait`.
        completed_workers, active_workers = ray.wait(active_workers)
        # If a failure occurs the ObjectRef will be marked as completed.
        # Calling ray.get will expose the failure as a RayActorError.
        for object_ref in completed_workers:
            try:
                ray.get(object_ref)
            # If an error occurs during the actor execution,
            # this error will get propagated as-is to the driver when you call ray.get().
            # For example, if a ValueError is raised in the actor method call,
            # this will be raised as a ValueError on the driver.
            # These exceptions will not be caught in this try-except clause
            except ray.exceptions.RayActorError as exc:
                _logger.error("Ray Actor error", exc)


if __name__ == "__main__":
    main()
