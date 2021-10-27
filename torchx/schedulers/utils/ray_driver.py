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
    def __init__(self, command):
        self.command = command

    def run_command(self):
        return subprocess.run(self.command)

class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)

def serialize(actor : RayActor) -> None:
    actor_json = json.dumps(actor, cls= EnhancedJSONEncoder)
    def write_json(actor, output_filename):
        with open(f"{output_filename}", 'w', encoding='utf-8') as f:
            json.dump([actor], f)
    write_json(actor_json, 'actor.json')

def load_actor_json(filename : str) -> List[Dict]:
    with open(filename) as f:
        actor = json.load(f)
        actor = json.loads(actor)
        return actor

if __name__ == "__main__":
    actor = RayActor("resnet", "echo hello")
    serialize(actor)
    logging.basicConfig(filename='driver.log', encoding='utf-8', level=logging.DEBUG)


    # On driver.py
    logging.debug("Reading actor.json")
    actors_dict = load_actor_json('actor.json')

    for actor_dict in actors_dict:

        bundle = {"CPU": actor_dict["num_cpus"], "GPU": actor_dict["num_gpus"]}
        bundles = [bundle] * actor_dict["num_replicas"]
        pg = ray.util.placement_group(bundles, strategy="SPREAD")

        logging.debug("Waiting for placement group to start.")
        ready, _ = ray.wait([pg.ready()], timeout=100)

        if ready:
            logging.debug("Placement group has started.")
            # ray.init(address="auto", namespace=actor_dict["name"])

            for key, value in actor_dict["env"]:
                os.environ[key] = value

            logging.debug("Environment variables set")

            logging.debug("Starting remote function")
        
        else:
            raise TimeoutError(
                "Placement group creation timed out. Make sure "
                "your cluster either has enough resources or use "
                "an autoscaling cluster. Current resources "
                "available: {}, resources requested by the "
                "placement group: {}".format(ray.available_resources(),
                                            pg.bundle_specs))

    # actor = CommandActor.options(placement_group=pg).remote(actor_dict["command"])
    # ray.get([actor.run_command.remote()])
    actors = [CommandActor.options(placement_group=pg).remote(actors_dict[i]["command"]) for i in range(len(actor_dict))]
    ray.get([a.run_command.remote() for a in actors])
