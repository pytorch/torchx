[![PyPI](https://img.shields.io/pypi/v/torchx)](https://pypi.org/project/torchx/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
![Tests](https://github.com/pytorch/torchx/actions/workflows/python-unittests.yaml/badge.svg)
![Lint](https://github.com/pytorch/torchx/actions/workflows/lint.yaml/badge.svg)
[![codecov](https://codecov.io/gh/pytorch/torchx/branch/main/graph/badge.svg?token=ceHHIm0hXy)](https://codecov.io/gh/pytorch/torchx)


# TorchX


TorchX is a universal job launcher for PyTorch applications.
TorchX is designed to have fast iteration time for training/research and support
for E2E production ML pipelines when you're ready.

TorchX currently supports:

* Kubernetes (EKS, GKE, AKS, etc)
* Slurm
* AWS Batch
* Docker
* Local
* Ray (prototype)

Need a scheduler not listed? [Let us know!](https://github.com/pytorch/torchx/issues?q=is%3Aopen+is%3Aissue+label%3Ascheduler-request)

## Quickstart

See the [quickstart guide](https://pytorch.org/torchx/latest/quickstart.html).

## Schedulers

[Supported schedulers]():



Planned Schedulers (need one? let us know!):

* Google AI Training Platform / TPUs #410
* Azure Batch #411


## Documentation

* [Stable Documentation](https://pytorch.org/torchx/latest/)
* [Nightly Documentation](https://pytorch.org/torchx/main/)

## Requirements

torchx:

* python3 (3.8+)
* [PyTorch](https://pytorch.org/get-started/locally/)
* optional: [Docker](https://docs.docker.com/get-docker/) (needed for docker based schedulers)

Certain schedulers may require scheduler specific requirements. See installation
for info.

## Installation

### Stable

```bash
# install torchx sdk and CLI -- minimum dependencies
pip install torchx

# install torchx sdk and CLI -- all dependencies
pip install "torchx[dev]"

# install torchx kubeflow pipelines (kfp) support
pip install "torchx[kfp]"

# install torchx Kubernetes / Volcano support
pip install "torchx[kubernetes]"

# install torchx Ray support
pip install "torchx[ray]"
```

### Nightly

```bash
# install torchx sdk and CLI
pip install torchx-nightly[dev]
```

### Source

```bash
# install torchx sdk and CLI from source
$ pip install -e git+https://github.com/pytorch/torchx.git#egg=torchx

# install extra dependencies
$ pip install -e git+https://github.com/pytorch/torchx.git#egg=torchx[dev]
```

### Docker

TorchX provides a docker container for using as as part of a TorchX role.

See: https://github.com/pytorch/torchx/pkgs/container/torchx

## Contributing

We welcome PRs! See the [CONTRIBUTING](CONTRIBUTING.md) file.

## License

TorchX is BSD licensed, as found in the [LICENSE](LICENSE) file.
