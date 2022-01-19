# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import importlib
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Tuple

import ray
from ray.train.utils import get_address_and_port
from ray.util.placement_group import PlacementGroup
from torchx.schedulers.ray.ray_common import RayActor

_logger: logging.Logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


@contextlib.contextmanager
def redirect_argv(args: List[str]):  # pyre-ignore[3]
    _argv = sys.argv[:]
    sys.argv = args
    yield
    sys.argv = _argv


@ray.remote
class CommandActor:  # pragma: no cover
    def __init__(self, command: str, env: Dict[str, str]) -> None:
        self.args: List[str] = command.split(" ")
        self.path: str = self.args[0]

        for k, v in env.items():
            os.environ[k] = v

    def exec_module(self) -> None:
        spec: Optional[
            importlib.machinery.ModuleSpec
        ] = importlib.util.spec_from_file_location("__main__", self.path)
        if spec:  # pragma: no cover
            train = importlib.util.module_from_spec(spec)
            with redirect_argv(self.args):
                spec.loader.exec_module(train)  # pyre-ignore[16]

    def get_actor_address_and_port(self) -> Tuple[str, int]:
        return get_address_and_port()

    def set_address_and_port(self, address: str, port: int) -> None:
        os.environ["MASTER_PORT"] = str(port)
        os.environ["MASTER_ADDR"] = address


def load_actor_json(filename: str) -> List[RayActor]:
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

        if ready:
            _logger.info("Placement group has started.")
            _logger.info("Starting remote function")
        else:  # pragma: no cover
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

    # 1. Create actors
    # 2. For each actor get rank 0 address and port
    # 3. Set address and port in command actor
    command_actors: List[CommandActor] = []
    # address, port = get_address_and_port()
    for i in range(len(actors)):
        world_size = actors[i].num_replicas
        actors_for_this_group = []

        for rank in range(world_size):

            # Environment variables for distributed training
            rank_env = {
                "WORLD_SIZE": str(world_size),
                "RANK": str(rank),
            }

            actor_and_rank_env = {**actors[i].env, **rank_env}

            actors_for_this_group.append(
                CommandActor.options(  # pyre-ignore[16]
                    placement_group=pgs[i],
                    num_cpus=actors[i].num_cpus,
                    num_gpus=actors[i].num_gpus,
                ).remote(actors[i].command, actor_and_rank_env)
            )

            rank_0_address, rank_0_port = ray.get(
                actors_for_this_group[0].get_actor_address_and_port.remote()
            )

            for actor in actors_for_this_group:
                ray.get(actor.set_address_and_port.remote(rank_0_address, rank_0_port))

            command_actors.extend(actors_for_this_group)

    return command_actors


if __name__ == "__main__":  # pragma: no cover
    _logger.debug("Reading actor.json")

    actors: List[RayActor] = load_actor_json("actors.json")
    os.remove("actors.json")

    _logger.debug("Creating Ray placement groups")
    ray.init(address="auto", namespace="torchx-ray")
    pgs: List[PlacementGroup] = create_placement_groups(actors)

    _logger.debug("Getting command actors")
    command_actors: List[CommandActor] = create_command_actors(actors, pgs)

    _logger.debug("Running Ray actors")
    active_workers = [  # pyre-ignore
        command_actor.exec_module.remote()  # pyre-ignore
        for command_actor in command_actors
    ]

    # Await return result of remote ray function
    while len(active_workers) > 0:
        completed_workers, active_workers = ray.wait(active_workers)
        # If a failure occurs the ObjectRef will be marked as completed.
        # Calling ray.get will expose the failure as a RayActorError.
        for object_ref in completed_workers:
            try:
                ray.get(object_ref)
                _logger.info("Ray remote function promise succesfully returned")

            # If an error occurs during the actor execution, this error will get propagated as-is to the driver when you call ray.get().
            # For example, if a ValueError is raised in the actor method call, this will be raised as a ValueError on the driver.
            # These exceptions will not be caught in this try-except clause
            except ray.exceptions.RayActorError as exc:
                _logger.info("Ray Actor Error")
                _logger.error("Ray Actor error", exc)
