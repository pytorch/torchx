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
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import ray
from ray.exceptions import RayActorError
from ray.train.utils import get_address_and_port
from ray.util.placement_group import PlacementGroup

if TYPE_CHECKING:
    from torchx.schedulers.ray.ray_common import (
        CommandActorScheduled,
        RayActor,
        TaskCompleted,
        TORCHX_RANK0_HOST,
    )

# Hack to make code work for tests as well as running ray job.
# For tests the `torchx.schedulers.ray.ray_common` import must be used
# For running ray jobs `ray_common` import must be used
try:
    # pyre-fixme[21]: Could not find a module corresponding to import `ray_common`.
    from ray_common import (
        CommandActorScheduled,
        RayActor,
        TaskCompleted,
        TORCHX_RANK0_HOST,
    )
except ModuleNotFoundError:
    from torchx.schedulers.ray.ray_common import (
        CommandActorScheduled,
        RayActor,
        TaskCompleted,
        TORCHX_RANK0_HOST,
    )

_logger: logging.Logger = logging.getLogger(__name__)
_logger.setLevel(logging.getLevelName(os.environ.get("LOGLEVEL", "INFO")))
logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))


@ray.remote
class CommandActor:  # pragma: no cover
    def __init__(self, cmd: List[str], env: Dict[str, str]) -> None:
        self.cmd: List[str] = cmd
        self.env: Dict[str, str] = env

    def exec_module(
        self, master_addr: str, master_port: int, actor_id: str
    ) -> TaskCompleted:
        """Execute a user script"""
        if master_addr is None or master_port is None:
            raise RuntimeError(
                "Either MASTER_ADDR or MASTER_PORT are not set. This is most likely bug in torchx"
                "Open issue at https://github.com/pytorch/torchx"
            )
        worker_evn = {}
        worker_evn.update(os.environ)
        worker_evn.update(self.env)
        worker_evn[TORCHX_RANK0_HOST] = master_addr
        popen = subprocess.Popen(self.cmd, env=worker_evn)

        returncode = popen.wait()
        _logger.info(f"Finished with code {returncode}")

        if returncode != 0:
            raise RuntimeError(f"exec_module failed with return code {returncode}")

        return TaskCompleted(actor_id)

    def schedule(self, actor_id: str) -> CommandActorScheduled:
        """Testing if a command actor is scheduled"""
        return CommandActorScheduled(actor_id)

    def get_actor_address_and_port(self) -> Tuple[str, int]:
        return get_address_and_port()


def load_actor_json(filename: str) -> List[RayActor]:
    """Loading replicas specifications from a JSON file"""
    with open(filename) as f:
        actors: List[RayActor] = []
        # Yes this is gross but it works
        actor_dict = json.load(f)
        actor_dict = json.loads(actor_dict)
        for actor in actor_dict:
            actors.append(RayActor(**actor))
        return actors


def create_placement_group_async(replicas: List[RayActor]) -> PlacementGroup:
    """return a placement group reference, the corresponding placement group could be scheduled or pending"""
    bundles = []
    for replica in replicas:
        bundles.append({"CPU": replica.num_cpus, "GPU": replica.num_gpus})

    pg = ray.util.placement_group(bundles, strategy="SPREAD")
    return pg


@dataclass
class ActorInfo:
    pg: PlacementGroup
    replica: RayActor
    actor: CommandActor


class RayDriver:
    def __init__(self, replicas: List[RayActor]) -> None:
        self.replicas = replicas
        self.master_node_id: Optional[str] = None
        self.rank_0_address: Optional[str] = None
        self.rank_0_port: Optional[int] = None
        self.min_nnodes, self.max_nnodes = parse_nnodes_rep(replicas)  # pyre-ignore

        self.placement_groups: List[PlacementGroup] = []
        self.actor_info_of_id: Dict[str, ActorInfo] = {}
        self.active_tasks: List["ray.ObjectRef"] = []

    def init_placement_groups(self) -> None:
        """Initialize all placement groups needed for this job"""
        replica_ix_of_pg: List[int] = [0] + list(
            range(self.min_nnodes, self.max_nnodes + 1)
        )
        # trace all the placement groups, {placemeng_group_reference: placement_group_index}
        self.placement_groups = [
            create_placement_group_async(
                self.replicas[replica_ix_of_pg[i] : replica_ix_of_pg[i + 1]]
            )
            for i in range(len(replica_ix_of_pg) - 1)
        ]

    def pop_actor_info(self, actor_id: str) -> ActorInfo:
        """Remove the info of  (dead) command actor"""
        return self.actor_info_of_id.pop(actor_id)

    def reschedule_actor(self, actor_id: str) -> None:
        """Rescheule a failed actor"""
        # pop the information of failed actor
        info = self.pop_actor_info(actor_id)
        _logger.info(f"Rescheduling actor {actor_id} to placement group {info.pg}")
        self.create_and_schedule_actor(info.pg, info.replica)

    def create_and_schedule_actor(self, pg: PlacementGroup, replica: RayActor) -> None:
        """create an command actor in the given placement group"""
        actor = CommandActor.options(  # pyre-ignore[16]
            placement_group=pg,
            num_cpus=replica.num_cpus,
            num_gpus=replica.num_gpus,
        ).remote(replica.command, replica.env)

        actor_id = actor._actor_id.hex()
        # launch a task to check if the actor is scheduled
        self.active_tasks.append(actor.schedule.remote(actor_id))
        self.actor_info_of_id[actor_id] = ActorInfo(
            actor=actor,
            pg=pg,
            replica=replica,
        )

    def place_command_actors(self) -> None:
        """Creating pending tasks to initialize command actors in placement groups"""
        pg_ix_of_replica: List[int] = [
            max(0, i - self.min_nnodes + 1) for i in range(len(self.replicas))
        ]  # the placement group indices for corresponding replicas
        for i in range(len(self.replicas)):
            pg_ix = pg_ix_of_replica[i]
            pg = self.placement_groups[pg_ix]  # find the created placement group
            replica = self.replicas[i]
            self.create_and_schedule_actor(pg, replica)

    def run(self) -> None:
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
                try:
                    result = ray.get(object_ref)  # pyre-ignore
                    if isinstance(result, CommandActorScheduled) and need_more_actors:
                        actor = self.actor_info_of_id[result.id].actor
                        if self.master_node_id is None:
                            # make this actor be the master node
                            self.master_node_id = result.id
                            self.rank_0_address, self.rank_0_port = ray.get(
                                actor.get_actor_address_and_port.remote()
                            )
                            self.active_tasks.append(
                                actor.exec_module.remote("localhost", 0, result.id)
                            )
                        else:
                            self.active_tasks.append(
                                actor.exec_module.remote(
                                    self.rank_0_address, self.rank_0_port, result.id
                                )
                            )
                        command_actors_count += 1
                    elif isinstance(result, TaskCompleted):
                        need_more_actors = False  # don't need more actors
                        command_actors_count -= 1  # 1 completed command actor
                        self.pop_actor_info(result.id)
                        if (
                            command_actors_count == 0
                        ):  # all the command actors have finished
                            break  # exit
                    else:
                        raise RuntimeError(
                            "Ray actor returns unkown type. This is most likely bug in torchx"
                            "Open issue at https://github.com/pytorch/torchx"
                        )
                except RayActorError as err:
                    # reschedule the failed command actor (node failure)
                    command_actors_count -= 1  # remove the failed actor
                    failed_actor_id: str = parse_actor_id_from_error(err)
                    _logger.info(
                        f"Node failure detected on command actor: {failed_actor_id}"
                    )
                    if failed_actor_id == self.master_node_id:
                        raise RuntimeError(
                            "Master node failed, cannot recover from master node failure"
                        )
                    self.reschedule_actor(failed_actor_id)


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


def parse_actor_id_from_error(err: RayActorError) -> str:
    msg = err.error_msg.split()
    try:
        id_ix = msg.index("actor_id:") + 1
    except ValueError:
        raise RuntimeError(
            "Experiencing a node failure,fault tolerance feature is not compatible with current ray version"
            "This is most likely bug in torchx"
            "Open issue at https://github.com/pytorch/torchx"
        )
    actor_id = msg[id_ix]
    return actor_id


def main() -> None:  # pragma: no cover
    actors: List[RayActor] = load_actor_json("actors.json")
    driver = RayDriver(actors)
    # pyre-fixme[16]: Module `worker` has no attribute `init`.
    ray.init(address="auto", namespace="torchx-ray")
    driver.init_placement_groups()
    _logger.info("Successfully created placement groups")
    driver.place_command_actors()
    _logger.info("Successfully placed command actors")
    _logger.info("Entering main loop, start executing the script on worker nodes")
    driver.run()


if __name__ == "__main__":
    main()
