import os
import json
import dataclasses
from dataclasses import dataclass, field
import ray
from typing import Any, Dict, Iterable, List, Optional, Set, Type
import logging
import subprocess


@dataclass
class RayActor:
    name: str
    command: str
    env: Dict[str, str] = field(default_factory=dict)
    num_replicas: int = 1
    num_cpus: int = 1
    num_gpus: int = 0
    memory_size: int = 1

@ray.remote
class CommandActor:
    def __init__(self, command, env):
        self.command = command
        self.env = env

    def run_command(self):
        return subprocess.run(self.command, env = self.env)

class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)

def serialize(actor : List[RayActor]) -> None:
    actor_json = json.dumps(actor, cls= EnhancedJSONEncoder)
    def write_json(actor, output_filename):
        with open(f"{output_filename}", 'w', encoding='utf-8') as f:
            json.dump(actor, f)
    write_json(actor_json, 'actor.json')

def load_actor_json(filename : str) -> List[Dict]:
    with open(filename) as f:
        actor = json.load(f)
        actor = json.loads(actor)
        return actor

if __name__ == "__main__":
    actor1 = RayActor("resnet", ["echo", "hello resnet"])
    actor2 = RayActor("bert", ["echo", "hello bert"])
    serialize([actor1, actor2])

    logging.basicConfig(filename='driver.log', encoding='utf-8', level=logging.DEBUG)


    # On driver.py
    logging.debug("Reading actor.json")
    actors_dict = load_actor_json('actor.json')
    pgs = []
    ray.init(address="auto", namespace="torchx-ray")

    for actor_dict in actors_dict:

        bundle = {"CPU": actor_dict["num_cpus"], "GPU": actor_dict["num_gpus"]}
        bundles = [bundle] * actor_dict["num_replicas"]
        pg = ray.util.placement_group(bundles, strategy="SPREAD")
        pgs.append(pg)

        logging.debug("Waiting for placement group to start.")
        ready = pg.wait(timeout_seconds=100)


        if ready:
            logging.debug("Placement group has started.")
            logging.debug("Starting remote function")
        
        else:
            raise TimeoutError(
                "Placement group creation timed out. Make sure "
                "your cluster either has enough resources or use "
                "an autoscaling cluster. Current resources "
                "available: {}, resources requested by the "
                "placement group: {}".format(ray.available_resources(),
                                            pg.bundle_specs))

    actors = [CommandActor.options(placement_group=pgs[i], num_cpus=actor_dict[i]["num_cpus"], num_gpus=actor_dict[i]["num_gpus"]).remote(actors_dict[i]["command"], actor_dict["env"]) for i in range(len(actors_dict) * actor_dict["num_replicas"])]
    ray.get([a.run_command.remote() for a in actors])
    unfinished = [a.run_command.remote() for a in actors]

    while len(unfinished) > 0:
        finished, unfinished = ray.wait(unfinished)
        # If a failure occurs the ObjectRef will be marked as finished.
        # Calling ray.get will expose the failure as a RayActorError.
        for object_ref in finished:
            try:
                status_code = ray.get(object_ref)
            except ray.RayActorError as exc:
                status_code = 1

            if status_code != 0:
                raise RuntimeError("Job failed")

