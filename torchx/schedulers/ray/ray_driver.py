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

from hashlib import new
import json
import logging
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import ray
from ray.exceptions import RayActorError
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
    def __init__(self, command: List[str], env: Dict[str, str], pg: PlacementGroup) -> None:
        self.cmd: List[str] = command
        self.env = env
        self.master_addr: Optional[str] = None
        self.master_port: Optional[int] = None
        self.pg: PlacementGroup = pg

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


def create_placement_group_async(replicas: List[RayActor]) -> PlacementGroup:
    # return a placement group reference, the corresponding placement group could be
    # scheduled or pending
    bundles = []
    for replica in replicas:
        bundles.append({"CPU": replica.num_cpus, "GPU": replica.num_gpus})

    pg = ray.util.placement_group(bundles, strategy="SPREAD")
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
        ).remote(replica.command, replica.env, pg)
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


def parse_actor_id_from_error(err: RayActorError) -> RayActor:
    msg = err.error_msg.split()
    id_ix = msg.index("actor_id:")+1
    actor_id = msg[id_ix]
    return actor_id


def main() -> None:  # pragma: no cover
    actors: List[RayActor] = load_actor_json("actors.json")
    min_nnodes, max_nnodes = parse_nnodes_rep(actors)
    actor_ixs = [0] + list(range(min_nnodes, max_nnodes + 1))
    # pyre-fixme[16]: Module `worker` has no attribute `init`.
    ray.init(address="auto", namespace="torchx-ray")

    placement_groups: List[PlacementGroup] = [
        create_placement_group_async(actors[actor_ixs[i] : actor_ixs[i + 1]])
        for i in range(len(actor_ixs) - 1)
    ]  # trace all the placement groups, {placemeng_group_reference: placement_group_index}
    # pyre-ignore: `ray._raylet.PlacementGroupID` is not defined as a type.
    pg_ids: Dict["ray._raylet.PlacementGroupID", int] = {
        placement_groups[i].id: (actor_ixs[i], actor_ixs[i + 1])
        for i in range(len(actor_ixs) - 1)
    }  # {pg_id: actor_index}
    active_tasks: List["ray.ObjectRef"] = [
        pg.ready() for pg in placement_groups
    ]  # tasks of creating all required placement groups

    active_actors: Dict[str, CommandActor] = {}
    need_more_actors: bool = True  # if need more actors
    command_actors_count: int = 0  # number of created command actors
    result: Optional[
        PlacementGroup
    ]  # result from a completed task, either a command execution result None or a created placement group
    # Await return result of remote ray function
    while len(active_tasks) > 0:
        _logger.info(f"running ray.wait on {active_tasks}")

        # ray.wait is partial waiting
        # pyre-fixme[16]: Module `worker` has no attribute `wait`.
        completed_tasks, active_tasks = ray.wait(active_tasks)
        # If a failure occurs the ObjectRef will be marked as completed.
        # Calling ray.get will expose the failure as a RayActorError.
        for object_ref in completed_tasks:
            # completed tasks contains two kinds of tasks:
            # 1) placement group creation; 2) command actor execution
            try:
                result = ray.get(object_ref)  # pyre-ignore
                if isinstance(result, PlacementGroup) and need_more_actors:
                    new_actors: List[CommandActor] = create_command_actors(
                        actors[
                            pg_ids[result.id][0] : pg_ids[result.id][1]
                        ],  # find the actor of a placement group based on pg_id
                        result,
                    )
                    for new_actor in new_actors:
                        active_actors[new_actor._actor_id.hex()] = new_actor
                        active_tasks.append(
                            new_actor.exec_module.remote()  # pyre-ignore
                        )  # add a new command actor execution task to the active tasks
                        command_actors_count += (
                            1  # monitor the number of active command actors
                        )
                else:
                    need_more_actors = False  # don't need more actors
                    command_actors_count -= 1  # 1 completed command actor
                    if (
                        command_actors_count == 0
                    ):  # all the command actors have finished
                        break  # exit
            except RayActorError as err:
                actor_id = parse_actor_id_from_error(err)
                actor = active_actors[actor_id]
                pg = actor.pg

if __name__ == "__main__":
    main()
