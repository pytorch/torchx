# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
ray_driver.py uses placement groups to manage command actors. each
placement group only holds one command actor. In both elastic and
non-elastic settings, we create all the placement groups at the beginning,
this step is non-blocking, since it doesn't wait all the placement
groups to be scheduled. When a placement group is scheduled successfully(
from pg.ready()), we allocate a new command actor in this placement group,
and let the actor execute the script. Once one of the command actor returns,
we do not need to create more command actors, and set `need_more_actors` flag
to False, and start waiting for all the command actors to complete. The number
of command actors are counted by `command_actors_count`.
"""

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


def create_placement_group_async(replicas: List[RayActor]) -> PlacementGroup:
    """return a placement group reference, the corresponding placement group could be scheduled or pending
    """
    bundles = []
    for replica in replicas:
        bundles.append({"CPU": replica.num_cpus, "GPU": replica.num_gpus})

    pg = ray.util.placement_group(bundles, strategy="SPREAD")
    return pg


class RayDriver:
    def __init__(self, actors: List[RayActor]):
        self.actors = actors
        self.rank_0_address: Optional[str] = None
        self.rank_0_port: Optional[int] = None
        min_nnodes, max_nnodes = parse_nnodes_rep(actors)
        self.actor_ixs = [0] + list(range(min_nnodes, max_nnodes + 1))  # helper for find placement groups

    def init_placement_groups(self):
        """Initialize all placement groups needed for this job"""
        # trace all the placement groups, {placemeng_group_reference: placement_group_index}
        self.placement_groups: List[PlacementGroup] = [
            create_placement_group_async(self.actors[self.actor_ixs[i] : self.actor_ixs[i + 1]])
                for i in range(len(self.actor_ixs) - 1)
        ]

        # the indices of actors in the placement group
        # pyre-ignore: `ray._raylet.PlacementGroupID` is not defined as a type.
        self.pg_ids: Dict["ray._raylet.PlacementGroupID", int] = {
            self.placement_groups[i].id: (self.actor_ixs[i], self.actor_ixs[i + 1])
                for i in range(len(self.actor_ixs) - 1)
        }

        # monitoring creation of the placement groups
        self.active_tasks: List["ray.ObjectRef"] = [
            pg.ready() for pg in self.placement_groups
        ]

    def create_command_actors(
        self, actors: List[RayActor], pg: PlacementGroup
    ) -> List[CommandActor]:
        """Create command actors in a give placement group"""
        cmd_actors: List[CommandActor] = []
        for _, replica in enumerate(actors):
            # Environment variables for distributed training
            actor = CommandActor.options(  # pyre-ignore[16]
                placement_group=pg,
                num_cpus=replica.num_cpus,
                num_gpus=replica.num_gpus,
            ).remote(replica.command, replica.env)
            cmd_actors.append(actor)

            if self.rank_0_address is None: 
                # make this actor the master node
                ray.get(actor.set_address_and_port.remote("localhost", 0))
                self.rank_0_address, self.rank_0_port = ray.get(
                    # pyre-ignore[16]
                    cmd_actors[0].get_actor_address_and_port.remote()
                )
            else:
                ray.get(actor.set_address_and_port.remote(self.rank_0_address, self.rank_0_port))
        return cmd_actors

    def get_actors_in_placement_group(self, id: "ray._raylet.PlacementGroupID") -> List[RayActor]:
        """Find the actors for a given placement group"""
        begin, end = self.pg_ids[id]
        return self.actors[begin: end]

    def run(self):
        result: Optional[PlacementGroup]  # execution result
        need_more_actors: bool = True  # if need more actors
        command_actors_count: int = 0  # number of created command actors
        # Await return result of remote ray function and initialize new command actors
        while len(self.active_tasks) > 0:
            _logger.info(f"running ray.wait on {self.active_tasks}")
            # ray.wait is partial waiting
            # pyre-fixme[16]: Module `worker` has no attribute `wait`.
            completed_tasks, self.active_tasks = ray.wait(self.active_tasks)
            # If a failure occurs the ObjectRef will be marked as completed.
            # Calling ray.get will expose the failure as a RayActorError.
            for object_ref in completed_tasks:
                # completed tasks contains two kinds of tasks:
                # 1) placement group creation; 2) command actor execution
                result = ray.get(object_ref)  # pyre-ignore
                if isinstance(result, PlacementGroup) and need_more_actors:
                    new_actors: List[CommandActor] = self.create_command_actors(
                        self.get_actors_in_placement_group(result.id),
                        result,
                    )
                    for new_actor in new_actors:
                        # add a new command actor execution task to the active tasks
                        self.active_tasks.append(
                            new_actor.exec_module.remote()  # pyre-ignore
                        )
                        # monitor the number of active command actors
                        command_actors_count += 1
                else:
                    need_more_actors = False  # don't need more actors
                    command_actors_count -= 1  # 1 completed command actor
                    if command_actors_count == 0:  # all the command actors have finished
                        break  # exit
    

def parse_nnodes_rep(actors: List[RayActor]) -> Tuple[int, int]:
    rep: Optional[str] = actors[0].nnodes_rep
    if rep is None:
        return len(actors), len(actors)
    if ":" in rep:
        min_nnodes, max_nnodes = rep.split(":")
        min_nnodes, max_nnodes = int(min_nnodes), int(max_nnodes)
    else:
        min_nnodes, max_nnodes = int(rep), int(rep)
    return min_nnodes, max_nnodes


def main() -> None:  # pragma: no cover
    actors: List[RayActor] = load_actor_json("actors.json")
    driver = RayDriver(actors)
    # pyre-fixme[16]: Module `worker` has no attribute `init`.
    ray.init(address="auto", namespace="torchx-ray")
    driver.init_placement_groups()
    driver.run()


if __name__ == "__main__":
    main()
