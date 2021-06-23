[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)![Tests](https://github.com/pytorch/torchx/actions/workflows/python-unittests.yaml/badge.svg)![Lint](https://github.com/pytorch/torchx/actions/workflows/lint.yaml/badge.svg)


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
```bash
# install torchx sdk and CLI
pip install torchx

# install torchx kubeflow pipelines (kfp) support
pip install "torchx[kfp]"
```

## Quickstart

See the [quickstart guide](https://pytorch.org/torchx/latest/quickstart.html).

## Contributing

We welcome PRs! See the [CONTRIBUTING](CONTRIBUTING.md) file.

## License

TorchX is BSD licensed, as found in the [LICENSE](LICENSE) file.
