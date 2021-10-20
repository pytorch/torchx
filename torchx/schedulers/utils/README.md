# Usage instructions

## User instructions

```bash
$ pip install torchx[ray]
$ torchx ray run --scheduler ray --cluster cluster.yaml

```

## Ray scheduler

Will run `os.system('ray exec python ray_driver.py actor.json')`
