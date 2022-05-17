# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function

# This script outputs relevant system environment info
# Run it with `python collect_env.py`.
import datetime
import locale
import os
import re
import subprocess
import sys
from os import PathLike
from typing import Callable, Dict, NamedTuple, Optional, Sequence, Tuple, Union


RunCallable = Callable[
    [
        Union[
            "PathLike[bytes]",
            "PathLike[str]",
            Sequence[Union["PathLike[bytes]", "PathLike[str]", bytes, str]],
            bytes,
            str,
        ]
    ],
    Tuple[int, str, str],
]


try:
    import torch

    TORCH_AVAILABLE = True
except (ImportError, NameError, AttributeError, OSError):
    TORCH_AVAILABLE = False

# System Environment Information
SystemEnv = NamedTuple(
    "SystemEnv",
    [
        ("torch_version", str),
        ("is_debug_build", str),
        ("cuda_compiled_version", str),
        ("gcc_version", str),
        ("clang_version", str),
        ("cmake_version", str),
        ("os", str),
        ("libc_version", str),
        ("python_version", str),
        ("python_platform", str),
        ("is_cuda_available", str),
        ("cuda_runtime_version", str),
        ("nvidia_driver_version", str),
        ("nvidia_gpu_models", str),
        ("pip_version", str),  # 'pip' or 'pip3'
        ("pip_packages", str),
        ("conda_packages", str),
        ("hip_compiled_version", str),
        ("caching_allocator_config", str),
        ("aws_version", str),
        ("gcp_version", str),
        ("azure_version", str),
        ("slurm_version", str),
        ("docker_version", str),
        ("kubectl_version", str),
    ],
)


def run(
    command: Union[
        "PathLike[bytes]",
        "PathLike[str]",
        Sequence[Union["PathLike[bytes]", "PathLike[str]", bytes, str]],
        bytes,
        str,
    ],
) -> Tuple[int, str, str]:
    """Returns (return-code, stdout, stderr)"""
    p = subprocess.Popen(
        args=command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    raw_output, raw_err = p.communicate()
    rc = p.returncode
    if get_platform() == "win32":
        enc = "oem"
    else:
        enc = locale.getpreferredencoding()
    output = raw_output.decode(enc)
    err = raw_err.decode(enc)
    return rc, output.strip(), err.strip()


def run_and_read_all(
    run_lambda: RunCallable,
    command: str,
) -> Optional[str]:
    """Runs command using run_lambda; reads and returns entire output if rc is 0"""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out


def run_and_parse_first_match(
    run_lambda: RunCallable, command: str, regex: str
) -> Optional[str]:
    """Runs command using run_lambda, returns the first regex match if it exists"""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    match = re.search(regex, out)
    if match is None:
        return None
    return match.group(1)


def run_and_return_first_line(run_lambda: RunCallable, command: str) -> Optional[str]:
    """Runs command using run_lambda and returns first line if output is not empty"""
    rc, out, _ = run_lambda(command)
    if rc != 0:
        return None
    return out.split("\n")[0]


def get_conda_packages(run_lambda: RunCallable) -> Optional[str]:
    conda = os.environ.get("CONDA_EXE", "conda")
    out = run_and_read_all(run_lambda, f"{conda} list")
    if out is None:
        return out

    return "\n".join(
        line
        for line in out.splitlines()
        if not line.startswith("#")
        and any(
            name in line
            for name in {
                "torch",
                "numpy",
                "cudatoolkit",
                "soumith",
                "mkl",
                "magma",
                "mkl",
            }
        )
    )


def get_gcc_version(run_lambda: RunCallable) -> Optional[str]:
    return run_and_parse_first_match(run_lambda, "gcc --version", r"gcc (.*)")


def get_clang_version(run_lambda: RunCallable) -> Optional[str]:
    return run_and_parse_first_match(
        run_lambda, "clang --version", r"clang version (.*)"
    )


def get_cmake_version(run_lambda: RunCallable) -> Optional[str]:
    return run_and_parse_first_match(run_lambda, "cmake --version", r"cmake (.*)")


def get_nvidia_driver_version(run_lambda: RunCallable) -> Optional[str]:
    if get_platform() == "darwin":
        cmd = "kextstat | grep -i cuda"
        return run_and_parse_first_match(
            run_lambda, cmd, r"com[.]nvidia[.]CUDA [(](.*?)[)]"
        )
    smi = get_nvidia_smi()
    return run_and_parse_first_match(run_lambda, smi, r"Driver Version: (.*?) ")


def get_gpu_info(run_lambda: RunCallable) -> Optional[str]:
    if get_platform() == "darwin" or (
        TORCH_AVAILABLE
        and hasattr(torch.version, "hip")
        and torch.version.hip is not None
    ):
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return torch.cuda.get_device_name(None)
        return None
    smi = get_nvidia_smi()
    uuid_regex = re.compile(r" \(UUID: .+?\)")
    rc, out, _ = run_lambda(smi + " -L")
    if rc != 0:
        return None
    # Anonymize GPUs by removing their UUID
    return re.sub(uuid_regex, "", out)


def get_running_cuda_version(run_lambda: RunCallable) -> Optional[str]:
    return run_and_parse_first_match(run_lambda, "nvcc --version", r"release .+ V(.*)")


def get_nvidia_smi() -> str:
    # Note: nvidia-smi is currently available only on Windows and Linux
    smi = "nvidia-smi"
    if get_platform() == "win32":
        system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
        program_files_root = os.environ.get("PROGRAMFILES", "C:\\Program Files")
        legacy_path = os.path.join(
            program_files_root, "NVIDIA Corporation", "NVSMI", smi
        )
        new_path = os.path.join(system_root, "System32", smi)
        smis = [new_path, legacy_path]
        for candidate_smi in smis:
            if os.path.exists(candidate_smi):
                smi = 'f"{candidate_smi}"'
                break
    return smi


def get_platform() -> str:
    if sys.platform.startswith("linux"):
        return "linux"
    elif sys.platform.startswith("win32"):
        return "win32"
    elif sys.platform.startswith("cygwin"):
        return "cygwin"
    elif sys.platform.startswith("darwin"):
        return "darwin"
    else:
        return sys.platform


def get_mac_version(run_lambda: RunCallable) -> Optional[str]:
    return run_and_parse_first_match(run_lambda, "sw_vers -productVersion", r"(.*)")


def get_windows_version(run_lambda: RunCallable) -> Optional[str]:
    system_root = os.environ.get("SYSTEMROOT", "C:\\Windows")
    wmic_cmd = os.path.join(system_root, "System32", "Wbem", "wmic")
    findstr_cmd = os.path.join(system_root, "System32", "findstr")
    return run_and_read_all(
        run_lambda, f"{wmic_cmd} os get Caption | {findstr_cmd} /v Caption"
    )


def get_lsb_version(run_lambda: RunCallable) -> Optional[str]:
    return run_and_parse_first_match(
        run_lambda, "lsb_release -a", r"Description:\t(.*)"
    )


def check_release_file(run_lambda: RunCallable) -> Optional[str]:
    return run_and_parse_first_match(
        run_lambda, "cat /etc/*-release", r'PRETTY_NAME="(.*)"'
    )


def get_os(run_lambda: RunCallable) -> Optional[str]:
    from platform import machine

    platform = get_platform()

    if platform == "win32" or platform == "cygwin":
        return get_windows_version(run_lambda)

    if platform == "darwin":
        version = get_mac_version(run_lambda)
        if version is None:
            return None
        return f"macOS {version} ({machine()})"

    if platform == "linux":
        # Ubuntu/Debian based
        desc = get_lsb_version(run_lambda)
        if desc is not None:
            return f"{desc} ({machine()})"

        # Try reading /etc/*-release
        desc = check_release_file(run_lambda)
        if desc is not None:
            return f"{desc} ({machine()})"

        return f"{platform} ({machine()})"

    # Unknown platform
    return platform


def get_python_platform() -> str:
    import platform

    return platform.platform()


def get_libc_version() -> str:
    import platform

    if get_platform() != "linux":
        return "N/A"
    return "-".join(platform.libc_ver())


def get_pip_packages(run_lambda: RunCallable) -> Tuple[str, str]:
    """Returns `pip list` output. Note: will also find conda-installed pytorch
    and numpy packages."""
    # People generally have `pip` as `pip` or `pip3`
    # But here it is incoved as `python -mpip`
    def run_with_pip(pip: str) -> str:
        out = run_and_read_all(run_lambda, f"{pip} list --format=freeze")
        assert out is not None
        return "\n".join(
            line
            for line in out.splitlines()
            if any(
                name in line
                for name in {
                    "torch",
                    "numpy",
                    "mypy",
                }
            )
        )

    pip_version = "pip3" if sys.version[0] == "3" else "pip"
    out = run_with_pip(sys.executable + " -mpip")

    return pip_version, out


def get_cachingallocator_config() -> str:
    ca_config = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
    return ca_config


def get_aws_version(run_lambda: RunCallable) -> Optional[str]:
    return run_and_read_all(run_lambda, "aws --version")


def get_gcp_version(run_lambda: RunCallable) -> Optional[str]:
    return run_and_parse_first_match(
        run_lambda, "gcloud --version", r"Google Cloud (.*)"
    )


def get_azure_version(run_lambda: RunCallable) -> Optional[str]:
    return run_and_parse_first_match(run_lambda, "az version", r"\"azure-cli\": (.*)")


def get_slurm_version(run_lambda: RunCallable) -> Optional[str]:
    return run_and_read_all(run_lambda, "slurmd --version")


def get_docker_version(run_lambda: RunCallable) -> Optional[str]:
    return run_and_parse_first_match(
        run_lambda, "docker --version", r"Docker version (.*)"
    )


def get_kubectl_version(run_lambda: RunCallable) -> Optional[str]:
    return run_and_parse_first_match(
        run_lambda, "kubectl version --client", r"Client Version: (.*)"
    )


def get_env_info() -> SystemEnv:
    run_lambda = run
    pip_version, pip_list_output = get_pip_packages(run_lambda)

    if TORCH_AVAILABLE:
        version_str = torch.__version__
        debug_mode_str = str(torch.version.debug)
        cuda_available_str = str(torch.cuda.is_available())
        cuda_version_str = torch.version.cuda
        if (
            not hasattr(torch.version, "hip") or torch.version.hip is None
        ):  # cuda version
            hip_compiled_version = "N/A"
        else:  # HIP version
            cfg = torch._C._show_config().split("\n")
            cuda_version_str = "N/A"
            hip_compiled_version = torch.version.hip
    else:
        version_str = debug_mode_str = cuda_available_str = cuda_version_str = "N/A"
        hip_compiled_version = "N/A"

    sys_version = sys.version.replace("\n", " ")

    return SystemEnv(
        torch_version=version_str,
        is_debug_build=debug_mode_str,
        python_version=f"{sys_version} ({sys.maxsize.bit_length() + 1}-bit runtime)",
        python_platform=get_python_platform(),
        is_cuda_available=cuda_available_str,
        cuda_compiled_version=cuda_version_str,
        cuda_runtime_version=str(get_running_cuda_version(run_lambda)),
        nvidia_gpu_models=str(get_gpu_info(run_lambda)),
        nvidia_driver_version=str(get_nvidia_driver_version(run_lambda)),
        hip_compiled_version=hip_compiled_version,
        pip_version=pip_version,
        pip_packages=pip_list_output,
        conda_packages=str(get_conda_packages(run_lambda)),
        os=str(get_os(run_lambda)),
        libc_version=get_libc_version(),
        gcc_version=str(get_gcc_version(run_lambda)),
        clang_version=str(get_clang_version(run_lambda)),
        cmake_version=str(get_cmake_version(run_lambda)),
        caching_allocator_config=get_cachingallocator_config(),
        aws_version=str(get_aws_version(run_lambda)),
        gcp_version=str(get_gcp_version(run_lambda)),
        azure_version=str(get_azure_version(run_lambda)),
        slurm_version=str(get_slurm_version(run_lambda)),
        docker_version=str(get_docker_version(run_lambda)),
        kubectl_version=str(get_kubectl_version(run_lambda)),
    )


env_info_fmt: str = """
PyTorch version: {torch_version}
Is debug build: {is_debug_build}
CUDA used to build PyTorch: {cuda_compiled_version}
ROCM used to build PyTorch: {hip_compiled_version}

OS: {os}
GCC version: {gcc_version}
Clang version: {clang_version}
CMake version: {cmake_version}
Libc version: {libc_version}

Python version: {python_version}
Python platform: {python_platform}
Is CUDA available: {is_cuda_available}
CUDA runtime version: {cuda_runtime_version}
GPU models and configuration: {nvidia_gpu_models}
Nvidia driver version: {nvidia_driver_version}

AWS: {aws_version}
GCP: {gcp_version}
Azure: {azure_version}
Slurm: {slurm_version}
Docker: {docker_version}
kubectl: {kubectl_version}

Versions of relevant libraries:
{pip_packages}
{conda_packages}
""".strip()


def pretty_str(envinfo: SystemEnv) -> str:
    def replace_nones(
        dct: Dict[str, str], replacement: str = "Could not collect"
    ) -> Dict[str, str]:
        for key in dct.keys():
            if dct[key] is not None:
                continue
            dct[key] = replacement
        return dct

    def replace_bools(
        dct: Dict[str, str], true: str = "Yes", false: str = "No"
    ) -> Dict[str, str]:
        for key in dct.keys():
            if dct[key] is True:
                dct[key] = true
            elif dct[key] is False:
                dct[key] = false
        return dct

    def prepend(text: str, tag: str = "[prepend]") -> str:
        lines = text.split("\n")
        updated_lines = [tag + line for line in lines]
        return "\n".join(updated_lines)

    def replace_if_empty(text: str, replacement: str = "No relevant packages") -> str:
        if text is not None and len(text) == 0:
            return replacement
        return text

    def maybe_start_on_next_line(string: str) -> str:
        # If `string` is multiline, prepend a \n to it.
        if string is not None and len(string.split("\n")) > 1:
            return f"\n{string}\n"
        return string

    mutable_dict = envinfo._asdict()

    # If nvidia_gpu_models is multiline, start on the next line
    mutable_dict["nvidia_gpu_models"] = maybe_start_on_next_line(
        envinfo.nvidia_gpu_models
    )

    # If the machine doesn't have CUDA, report some fields as 'No CUDA'
    dynamic_cuda_fields = [
        "cuda_runtime_version",
        "nvidia_gpu_models",
        "nvidia_driver_version",
    ]
    all_cuda_fields = dynamic_cuda_fields
    all_dynamic_cuda_fields_missing = all(
        mutable_dict[field] is None for field in dynamic_cuda_fields
    )
    if (
        TORCH_AVAILABLE
        and not torch.cuda.is_available()
        and all_dynamic_cuda_fields_missing
    ):
        for field in all_cuda_fields:
            mutable_dict[field] = "No CUDA"
        if envinfo.cuda_compiled_version is None:
            mutable_dict["cuda_compiled_version"] = "None"

    # Replace True with Yes, False with No
    mutable_dict = replace_bools(mutable_dict)

    # Replace all None objects with 'Could not collect'
    mutable_dict = replace_nones(mutable_dict)

    # If either of these are '', replace with 'No relevant packages'
    mutable_dict["pip_packages"] = replace_if_empty(mutable_dict["pip_packages"])
    mutable_dict["conda_packages"] = replace_if_empty(mutable_dict["conda_packages"])

    # Tag conda and pip packages with a prefix
    # If they were previously None, they'll show up as ie '[conda] Could not collect'
    if mutable_dict["pip_packages"]:
        mutable_dict["pip_packages"] = prepend(
            mutable_dict["pip_packages"], "[{}] ".format(envinfo.pip_version)
        )
    if mutable_dict["conda_packages"]:
        mutable_dict["conda_packages"] = prepend(
            mutable_dict["conda_packages"], "[conda] "
        )
    return env_info_fmt.format(**mutable_dict)


def get_pretty_env_info() -> str:
    return pretty_str(get_env_info())


def main() -> None:
    print("Collecting environment information...")
    output = get_pretty_env_info()
    print(output)

    if (
        TORCH_AVAILABLE
        and hasattr(torch, "utils")
        and hasattr(torch.utils, "_crash_handler")
    ):
        minidump_dir = torch.utils._crash_handler.DEFAULT_MINIDUMP_DIR
        if sys.platform == "linux" and os.path.exists(minidump_dir):
            dumps = [
                os.path.join(minidump_dir, dump) for dump in os.listdir(minidump_dir)
            ]
            latest = max(dumps, key=os.path.getctime)
            ctime = os.path.getctime(latest)
            creation_time = datetime.datetime.fromtimestamp(ctime).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            msg = (
                f"\n*** Detected a minidump at {latest} created on {creation_time}, "
                + "if this is related to your bug please include it when you file a report ***"
            )
            print(msg, file=sys.stderr)


if __name__ == "__main__":
    main()
