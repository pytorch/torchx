# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import json
import os
import subprocess
from typing import Any, Dict, Iterable, List, Optional, Set, Type

import ray

from .ray_common import RayActor


@ray.remote
class CommandActor:
    def __init__(self, command: str, env: Dict[str, str]):
        self.args = command.split(" ")
        self.path = self.args[0]

        # Set up the environment variables to be passed to the user's command.
        self.env = os.environ.copy()
        self.env.update(env)

    def run_command(self) -> None:
        os.execve(self.path, self.args, self.env)


def load_actor_json(filename: str) -> List[RayActor]:
    with open(filename) as f:
        actors = []
        for actor_dict in json.load(f):
            actors.append(RayActor(**actor_dict))

        return actors


def _main(job_id=None):
    print("Reading actor.json")
    actors = load_actor_json("actors.json")
    pgs = []
    ray.init(address="auto", namespace="torchx-ray")

    for actor in actors:

        bundle = {"CPU": actor.num_cpus, "GPU": actor.num_gpus}
        bundles = [bundle] * actor.num_replicas
        pg = ray.util.placement_group(bundles, strategy="SPREAD")
        pgs.append(pg)

        print("Waiting for placement group to start.")
        ready = pg.wait(timeout_seconds=100)

        if ready:
            print("Placement group has started.")
            print("Starting remote function")

        else:
            raise TimeoutError(
                "Placement group creation timed out. Make sure "
                "your cluster either has enough resources or use "
                "an autoscaling cluster. Current resources "
                "available: {}, resources requested by the "
                "placement group: {}".format(ray.available_resources(), pg.bundle_specs)
            )

    for i in range(len(actors)):
        for _ in range(actors[i].num_replicas):
            actors.append(
                CommandActor.options(
                    placement_group=pgs[i],
                    num_cpus=actors[i].num_cpus,
                    num_gpus=actors[i].num_gpus,
                ).remote(actors[i].command, actors[i].env)
            )

    unfinished = [actor.run_command.remote() for actor in actors]

    while len(unfinished) > 0:
        finished, unfinished = ray.wait(unfinished)
        # If a failure occurs the ObjectRef will be marked as finished.
        # Calling ray.get will expose the failure as a RayActorError.
        for object_ref in finished:
            try:
                ray.get(object_ref)

            except ray.RayActorError as exc:
                status_code = 1

            if status_code != 0:
                raise RuntimeError("Job failed")
