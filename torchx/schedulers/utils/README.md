# Usage instructions

## User instructions

```bash
$ pip install torchx[ray]
$ git clone https://github.com/ray-project/ray.git
$ torchx ray run --scheduler ray --cluster ray/python/ray/autoscaler/aws/example-full.yaml

```

## Ray scheduler

Will run `os.system('ray exec python ray_driver.py actor.json')`
