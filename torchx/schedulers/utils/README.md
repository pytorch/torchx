# Usage instructions

## User instructions

```bash
$ pip install torchx[ray]
$ git clone https://github.com/ray-project/ray.git
$ ray up ray/python/ray/autoscaler/aws/example-full.yaml
$ torchx ray run --scheduler ray 

```

## Ray scheduler

Will run `os.system('ray exec python ray_driver.py actor.json')`
