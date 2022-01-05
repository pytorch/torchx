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
from typing import Dict, List, Optional, Any

import ray
from ray.train.utils import get_address_and_port
from ray.util.placement_group import PlacementGroup
from torchx.schedulers.ray.ray_common import RayActor

_logger: logging.Logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


@contextlib.contextmanager
def redirect_argv(args : List[str]): # pyre-ignore[3]
    _argv = sys.argv[:]
    sys.argv = args
    yield
    sys.argv = _argv


@ray.remote
class CommandActor:
    def __init__(self, command: str, env: Dict[str, str]) -> None:
        self.args: List[str] = command.split(" ")
        self.path: str = self.args[0]

        for k, v in env.items():
            os.environ[k] = v

    def run_command(self) -> None:
        spec: Optional[
            importlib.machinery.ModuleSpec
        ] = importlib.util.spec_from_file_location("__main__", self.path)
        if spec:
            train = importlib.util.module_from_spec(spec)
            with redirect_argv(self.args):
                spec.loader.exec_module(train) # pyre-ignore[16]


def load_actor_json(filename: str) -> List[RayActor]:  # pyre-ignore[11]
    with open(filename) as f:
        actors: List[RayActor] = []
        # Yes this is gross but it works
        actor_dict = json.load(f)
        actor_dict = json.loads(actor_dict)
        for actor in actor_dict:
            actors.append(RayActor(**actor))
        return actors

def create_placement_groups(actors : List[RayActor]) -> List[PlacementGroup]:
    pgs : List[PlacementGroup] = []
    for actor in actors:
        bundle = {"CPU": actor.num_cpus, "GPU": actor.num_gpus}
        bundles = [bundle] * actor.num_replicas
        pg = ray.util.placement_group(bundles, strategy="SPREAD")
        pgs.append(pg)

        _logger.info("Waiting for placement group to start.")
        ready = pg.wait(timeout_seconds=100)

        if ready:
            _logger.info("Placement group has started.")
            _logger.info("Starting remote function")
        else:
            raise TimeoutError(
                "Placement group creation timed out. Make sure "
                "your cluster either has enough resources or use "
                "an autoscaling cluster. Current resources "
                "available: {}, resources requested by the "
                "placement group: {}".format(ray.available_resources(), pg.bundle_specs)
            )           
    return pgs

def create_command_actors(actors : List[RayActor], pgs : List[PlacementGroup]) -> List[CommandActor]:
    command_actors: List[CommandActor] = []
    address, port = get_address_and_port() # pyre-ignore[5]
    for i in range(len(actors)):
        world_size = actors[i].num_replicas

        for rank in range(world_size):

            # Environment variables for distributed training
            rank_env = {
                "WORLD_SIZE": str(world_size),
                "MASTER_PORT": str(port),
                "MASTER_ADDR": address,
                "RANK": str(rank),
            }

            actor_and_rank_env = {**actors[i].env, **rank_env}

            command_actors.append(
                CommandActor.options( # pyre-ignore[16]
                    placement_group=pgs[i],
                    num_cpus=actors[i].num_cpus,
                    num_gpus=actors[i].num_gpus,
                ).remote(actors[i].command, actor_and_rank_env) # pyre-ignore[16]
            )
    return command_actors


if __name__ == "__main__": # pragma: no cover
    _logger.info("Reading actor.json")
    actors : List[RayActor] = load_actor_json("actors.json")

    _logger.info("Creating Ray placement groups")
    ray.init(address="auto", namespace="torchx-ray")
    pgs : List[PlacementGroup] = create_placement_groups(actors)

    _logger.info("Getting command actors")
    command_actors : List[CommandActor] = create_command_actors(actors, pgs)

    _logger.info("Running Ray actors")
    unfinished : List[Any] = [command_actor.run_command.remote() for command_actor in command_actors] # pyre-ignore[16] 

    # Await return result of remote ray function
    while len(unfinished) > 0:
        finished, unfinished = ray.wait(unfinished) # pyre-ignore[5]
        # If a failure occurs the ObjectRef will be marked as finished.
        # Calling ray.get will expose the failure as a RayActorError.
        for object_ref in finished:
            try:
                ray.get(object_ref)
                _logger.info(f"Ray remote function promise succesfully returned")
            except ray.exceptions.RayActorError as exc:
                _logger.info("Ray Actor Error")
