import os
import json
import dataclasses
from dataclasses import dataclass, field
import ray
from typing import Any, Dict, Iterable, List, Optional, Set, Type
import logger


@dataclass
class RayActor:
    name: str
    command: str
    env: Dict[str, str] = field(default_factory=dict)
    num_replicas: int = 1
    num_cpus: int = 1
    num_gpus: int = 1
    memory_size: int = 1

class EnhancedJSONEncoder(json.JSONEncoder):
        def default(self, o):
            if dataclasses.is_dataclass(o):
                return dataclasses.asdict(o)
            return super().default(o)

def serialize(actor : RayActor) -> None:
    actor_json = json.dumps(actor, cls= EnhancedJSONEncoder)
    def write_json(actor, output_filename):
        with open(f"{output_filename}", 'w', encoding='utf-8') as f:
            json.dump(actor, f, ensure_ascii=False, indent=4)
    write_json(actor_json, 'actor.json')

def load_actor_json(filename : str) -> List[Dict]:
    with open(filename) as f:
        actor = json.load(f)
        actor = json.loads(actor)
        return actor

if __name__ == "__main__":
    # 1. Load json actor
    # 2. Create placement groups per actor
    # 3. Start Ray actor in each placement group 
    
    # On ray_scheduler.py
    actor = RayActor("ray", "up")
    serialize(actor)

    # On driver.py
    logger("Reading actor.json")
    actors_dict = load_actor_json('actor.json')

    for actor_dict in actors_dict:

        bundle = {"CPU": actor_dict["num_cpus"], "GPU": actor_dict["num_gpus"]}
        bundles = [bundle] * actor_dict["num_workers"]
        pg = ray.util.placement_group(bundles, strategy="SPREAD")

        logger.debug("Waiting for placement group to start.")
        ready, _ = ray.wait([pg.ready()], timeout=100)

        if ready:
            logger.debug("Placement group has started.")
            ray.init()

            for key, value in actor_dict["env"]:
                os.environ[key] = value

            logger.debug("Environment variables set")

            logger.debug("Starting remote function")

            ## TODO: Not sure about this part
            ## Do we need to introspect the script to get the nn.module?
            RemoteNetwork = ray.remote(actor_dict["command"])
            ray.get([NetworkActor.train.remote()])
            logger.debug("Remote function is running")

        else:
            raise TimeoutError(
                "Placement group creation timed out. Make sure "
                "your cluster either has enough resources or use "
                "an autoscaling cluster. Current resources "
                "available: {}, resources requested by the "
                "placement group: {}".format(ray.available_resources(),
                                            pg.bundle_specs))