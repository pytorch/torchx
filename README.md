[![PyPI](https://img.shields.io/pypi/v/torchx)](https://pypi.org/project/torchx/)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
![Tests](https://github.com/pytorch/torchx/actions/workflows/python-unittests.yaml/badge.svg)
![Lint](https://github.com/pytorch/torchx/actions/workflows/lint.yaml/badge.svg)
[![codecov](https://codecov.io/gh/pytorch/torchx/branch/main/graph/badge.svg?token=ceHHIm0hXy)](https://codecov.io/gh/pytorch/torchx)


# TorchX


TorchX is a library containing standard DSLs for authoring and running PyTorch
related components for an E2E production ML pipeline.

For the latest documentation, please refer to our [website](https://pytorch.org/torchx).


## Requirements
TorchX SDK (torchx):
* python3 (3.8+)
* torch

TorchX Kubeflow Pipelines Support (torchx-kfp):
* torchx
* kfp

## Installation

### Release

```bash
# install torchx sdk and CLI -- minimum dependencies
pip install torchx

# install torchx sdk and CLI -- all dependencies
pip install "torchx[dev]"

# install torchx kubeflow pipelines (kfp) support
pip install "torchx[kfp]"

# install torchx Kubernetes / Volcano support
pip install "torchx[kubernetes]"
```

### Nightly

```bash
# install torchx sdk and CLI
pip install torchx-nightly
```

### Source

```bash
# install torchx sdk and CLI from source
$ pip install -e git+https://github.com/pytorch/torchx.git#egg=torchx

# install extra dependencies
$ pip install -e git+https://github.com/pytorch/torchx.git#egg=torchx[kubernetes]
```

## Quickstart

See the [quickstart guide](https://pytorch.org/torchx/latest/quickstart.html).

## Contributing

We welcome PRs! See the [CONTRIBUTING](CONTRIBUTING.md) file.

## License

TorchX is BSD licensed, as found in the [LICENSE](LICENSE) file.
