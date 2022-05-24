# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This script uses https://raw.githubusercontent.com/pytorch/pytorch/master/torch/utils/collect_env.py
# and collects additional information on top of it to output relevant system
# environment info.
# Run it with `python collect_env.py`.
import re
import subprocess
import sys
import tempfile
from os import getenv
from os.path import exists
from typing import Optional, Tuple
from urllib import request

PYTORCH_COLLECT_ENV_URL = "https://raw.githubusercontent.com/pytorch/pytorch/master/torch/utils/collect_env.py"
TORCHX_PACKAGES = (
    "https://raw.githubusercontent.com/pytorch/torchx/main/dev-requirements.txt"
)


def run(
    command: str, filter_output_regexp: Optional[str] = None
) -> Optional[Tuple[int, bytes, bytes]]:
    """Returns (return-code, stdout, stderr)"""
    p = subprocess.Popen(
        args=command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    raw_output, raw_err = p.communicate()
    raw_output, raw_err = raw_output.strip().decode("utf-8"), raw_err.strip().decode(
        "utf-8"
    )
    rc = p.returncode
    if rc != 0:
        return None

    if filter_output_regexp:
        match = re.search(filter_output_regexp, raw_output)
        if match is None:
            return None
        return match.group(1)

    return rc, raw_output, raw_err


def get_pip_packages() -> str:
    """Returns versions of packages that match torchx dev requirements"""
    user_packages = subprocess.run(
        f"{sys.executable + ' -mpip'} list --format=freeze",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        shell=True,
    )

    torchx_packages, _ = request.urlretrieve(TORCHX_PACKAGES)
    with open(torchx_packages, "r") as packages:
        torchx_deps = [
            re.split(r"==|>=|<=|!=|!=|===|<|>", package.strip())[0]
            for package in packages.readlines()
            if package.strip() and not package.startswith("#")
        ]
        assert torchx_deps is not None

    user_deps = [
        re.split(r"==|>=|<=|!=|!=|===|<|>", line)
        for line in user_packages.stdout.decode("utf-8").splitlines()
    ]

    return "\n".join(
        f"{udeps[0]}:{udeps[1]}"
        for udeps in user_deps
        if any(tdeps in udeps[0] for tdeps in torchx_deps)
    )


def get_torchx_config() -> str:
    torchxconfig = None
    if exists(".torchxconfig"):
        torchxconfig = ".torchxconfig"
    elif exists(f"{getenv('HOME')}/.torchxconfig"):
        torchxconfig = f"{getenv('HOME')}/.torchxconfig"
    else:
        return "N/A"

    with open(torchxconfig, "r") as f:
        return f.read()


def run_pytorch_collect_env() -> Tuple[int, bytes]:
    with tempfile.NamedTemporaryFile(delete=True, suffix=".py") as temp:
        request.urlretrieve(PYTORCH_COLLECT_ENV_URL, temp.name)
        out = subprocess.run(
            f"{sys.executable} {temp.name}", stderr=subprocess.PIPE, shell=True
        )
        return out.returncode, out.stderr


def get_cli_info() -> None:
    print(f"AWS CLI: {get_aws_version()}")
    print(f"gCloud CLI: {get_gcp_version()}")
    print(f"AZ CLI: {get_azure_version()}")
    print(f"Slurm: {get_slurm_version()}")
    print(f"Docker: {get_docker_version()}")
    print(f"kubectl: {get_kubectl_version()}")


def get_aws_version() -> Optional[str]:
    result = run("aws --version")
    if result:
        return result[1]


def get_gcp_version() -> Optional[str]:
    return run("gcloud --version", r"Google Cloud (.*)")


def get_azure_version() -> Optional[str]:
    return run("az version", r"\"azure-cli\": (.*)")


def get_slurm_version() -> Optional[str]:
    result = run("slurmd --version")
    if result:
        return result[1]


def get_docker_version() -> Optional[str]:
    return run("docker --version", r"Docker version (.*)")


def get_kubectl_version() -> Optional[str]:
    return run("kubectl version --client", r"Client Version: (.*)")


def main() -> None:
    status = run_pytorch_collect_env()
    if status[0] != 0:
        print(f"Could not run Pytorch collect_env script: {status[1]}")

    print("\nVersions of CLIs:")
    get_cli_info()

    print("\ntorchx dev package versions:")
    print(get_pip_packages())

    print("\ntorchx config:")
    print(f"{get_torchx_config()}")


if __name__ == "__main__":
    main()
