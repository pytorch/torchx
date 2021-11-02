from dataclasses import dataclass, field

@dataclass
class RayActor:
    name: str
    command: str
    env: Dict[str, str] = field(default_factory=dict)
    num_replicas: int = 1
    num_cpus: int = 1
    num_gpus: int = 0
    memory_size: int = 1