import json
from json import JSONEncoder
import dataclasses
from dataclasses import dataclass, field
import ray
from typing import Any, Dict, Iterable, List, Optional, Set, Type


@dataclass
class RayActor:
    name: str
    command: str
    env: Dict[str, str] = field(default_factory=dict)
    num_replicas: int = 1
    num_cpus: int = -1
    num_gpus: int = -1
    memory_size: int = -1

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

def load_actor_json(filename : str) -> Dict:
    with open(filename) as f:
        actor = json.load(f)
        actor = json.loads(actor)
        return actor

if __name__ == "__main__":
    actor = RayActor("ray", "up")
    serialize(actor)
    actor_dict = load_actor_json('actor.json')

    # Untested below
    bundle1 = {"GPU": actor_dict[num_gpus]}
    bundle2 = {"extra_resource": actor_dict[num_replicas]}
    bundle3 = {"CPU": actor_dict[num_cpus]}
    # What happens with name, command and memory size

    # Is STRICK_PACK the correct strategy?
    pg = placement_group([bundle1, bundle2, bundle3], strategy="STRICT_PACK")    