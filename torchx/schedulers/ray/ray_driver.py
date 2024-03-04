# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
We use placement groups to reserve resources in the ray cluster, it
ensure that a job will not lose the resources it used to have before
the job is finished. The deadlock situtation while launch multiple jobs at the
same time is avoided by create a big placement group that contains the minimum
required command actors for the job. Once the placement groups are created(may
not be scheduled on a physical node yet), then we schedule command actors to
the corresponding placement group, each actor is associated with a placement
group which hold the resource the acotr needs. Each time a placement group successfully
acquired the resources from the ray cluster, the actor scheduled to this placement group
will be executed. Command actors are state machines their behavior is defined by the
_step function, this give more flexibility to us if we want to bette handle the
node failures.
"""
import json
import logging
import os
import socket
import subprocess
import sys

from contextlib import closing
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING

import ray
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


@dataclass
class RayResult:
    id: str


class TaskCompleted(RayResult):
    pass


class CommandActorScheduled(RayResult):
    pass


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
        addr = ray.util.get_node_ip_address()
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(("", 0))
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            port = s.getsockname()[1]
        return addr, port


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
    """Used to store the information for restoring a failed command actor"""

    pg: PlacementGroup
    replica: RayActor
    actor: CommandActor


class RayDriver:
    def __init__(self, replicas: List[RayActor]) -> None:
        self.replicas = replicas
        self.master_node_id: Optional[str] = None  # the actor id of the master node
        self.rank_0_address: Optional[str] = None
        self.rank_0_port: Optional[int] = None
        self.max_replicas: int = len(replicas)
        self.min_replicas: int
        if replicas[0].min_replicas is None:
            self.min_replicas = self.max_replicas
        else:
            self.min_replicas = replicas[0].min_replicas  # pyre-ignore[8]

        self.placement_groups: List[PlacementGroup] = (
            []
        )  # all the placement groups, shall never change
        self.actor_info_of_id: Dict[str, ActorInfo] = (
            {}
        )  # store the info used to recover an actor
        self.active_tasks: List["ray.ObjectRef"] = []  # list of active tasks

        self.terminating: bool = False  # if the job has finished and being terminated
        self.command_actors_count: int = 0  # number of created command actors

    def init_placement_groups(self) -> None:
        """Initialize all placement groups needed for this job"""
        # find the actor specifications of a given placement group
        replica_ix_of_pg: List[int] = [0] + list(
            range(
                self.min_replicas,
                self.max_replicas + 1,
            )
        )
        # create all the placement groups
        initial_group = create_placement_group_async(
            self.replicas[replica_ix_of_pg[0] : replica_ix_of_pg[1]]
        )
        _logger.info("Waiting for minimum placement group to start.")
        ready = initial_group.wait(100)
        if not ready:  # pragma: no cover
            raise TimeoutError(
                "Placement group creation timed out. Make sure "
                "your cluster either has enough resources or use "
                "an autoscaling cluster. Current resources "
                "available: {}, resources requested by the "
                "placement group: {}".format(
                    ray.available_resources(), initial_group.bundle_specs
                )
            )
        self.placement_groups.append(initial_group)
        for i in range(1, len(replica_ix_of_pg) - 1):
            self.placement_groups.append(
                create_placement_group_async(
                    self.replicas[replica_ix_of_pg[i] : replica_ix_of_pg[i + 1]]
                )
            )

    def pop_actor_info(self, actor_id: str) -> ActorInfo:
        """Remove and return the info of a dead command actor"""
        return self.actor_info_of_id.pop(actor_id)

    def create_and_schedule_actor(self, pg: PlacementGroup, replica: RayActor) -> None:
        """create an command actor in the given placement group"""
        # create the command actor
        actor = CommandActor.options(  # pyre-ignore[16]
            placement_group=pg,
            num_cpus=replica.num_cpus,
            num_gpus=replica.num_gpus,
        ).remote(replica.command, replica.env)

        # get the actor id of the created actor
        actor_id = actor._actor_id.hex()
        # launch a task to check if the actor is scheduled
        self.active_tasks.append(actor.schedule.remote(actor_id))
        # save the actor info for recovering from node failures
        self.actor_info_of_id[actor_id] = ActorInfo(
            actor=actor,
            pg=pg,
            replica=replica,
        )

    def place_command_actors(self) -> None:
        """Creating all command actors in all placement groups"""
        # find the placement group index for a replica(actor's specification)
        pg_ix_of_replica: List[int] = [
            max(0, i - self.min_replicas + 1) for i in range(len(self.replicas))
        ]
        # create the actors
        for i in range(len(self.replicas)):
            pg_ix = pg_ix_of_replica[i]
            pg = self.placement_groups[pg_ix]  # find the created placement group
            replica = self.replicas[i]
            self.create_and_schedule_actor(pg, replica)

    def _step(self) -> bool:
        """Handling command actor's return"""
        result: RayResult  # execution result
        _logger.info(f"running ray.wait on {self.active_tasks}")
        # ray.wait is partial waiting
        completed_tasks, self.active_tasks = ray.wait(self.active_tasks)
        # If a failure occurs the ObjectRef will be marked as completed.
        # Calling ray.get will expose the failure as a RayActorError.
        for object_ref in completed_tasks:
            result = ray.get(object_ref)
            if isinstance(result, CommandActorScheduled):
                if not self.terminating:
                    actor = self.actor_info_of_id[result.id].actor
                    if self.master_node_id is None:
                        # make this actor be the master node
                        self.master_node_id = result.id
                        self.rank_0_address, self.rank_0_port = ray.get(
                            actor.get_actor_address_and_port.remote()  # pyre-ignore
                        )
                        self.active_tasks.append(
                            actor.exec_module.remote(  # pyre-ignore
                                "localhost", 0, result.id
                            )
                        )
                    else:
                        self.active_tasks.append(
                            actor.exec_module.remote(
                                self.rank_0_address, self.rank_0_port, result.id
                            )
                        )
                    self.command_actors_count += 1
            elif isinstance(result, TaskCompleted):
                self.terminating = (
                    True  # terminating the job, wait for all actors to finish
                )
                self.command_actors_count -= 1  # 1 completed command actor
                self.pop_actor_info(result.id)
                if (
                    self.command_actors_count == 0
                ):  # all the command actors have finished
                    return True  # is terminal
            else:
                raise RuntimeError(
                    f"Ray actor returns unknown type {type(result)}"
                    "This is most likely bug in torchx"
                    "Open issue at https://github.com/pytorch/torchx"
                )
        return False

    def run(self) -> None:
        """This is the main loop the ray driver, it executes the user script on the scheduled nodes,
        and restart the failed nodes(node failures). The loop ends when all the actors that joining
        the job exits."""
        self.terminating = False
        self.command_actors_count = 0
        # Await return result of remote ray function and initialize new command actors
        while len(self.active_tasks) > 0:
            terminal = self._step()
            if terminal:
                break


def main() -> None:  # pragma: no cover
    actors: List[RayActor] = load_actor_json("actors.json")
    driver = RayDriver(actors)
    ray.init(address="auto", namespace="torchx-ray")
    driver.init_placement_groups()
    _logger.info("Successfully created placement groups")
    driver.place_command_actors()
    _logger.info("Successfully placed command actors")
    _logger.info("Entering main loop, start executing the script on worker nodes")
    driver.run()


if __name__ == "__main__":
    main()
