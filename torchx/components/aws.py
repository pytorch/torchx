
"""
This module contains a custom ``spmd`` component to use with AWS Batch.
It differs from the default ``spmd`` component in that it has hard-coded certain configurations (env vars, torchrun parameters etc).
that makes distributed (DDP or FSDP) scripts run smoothly on AWS Batch.

Notable differences:

#. Uses ``static`` rendezvous backend to pin AWS Batch assigned node 0 for elastic agent 0
#. Sets ``NCCL_SOCKET_IFNAME=eth,ens`` to be compatible with the network iface names on AWS instances when 
    provisioned with the DeepLearning AMI (Ubuntu and AmazonLinux 2).
#. Sets ``FI_PROVIDER=efa`` and ``FI_EFA_USE_DEVICE_RDMA=1`` to let PyTorch (via NCCL) run collectives over EFA (AWS's proprietary NIC).


For usage run:

.. code-block:: shell-session

    $ torchx run amzn.spmd --help
    torchx 2023-02-23 22:41:26 INFO     loaded configs from /home/ubuntu/workspace/torchx/.torchxconfig
    usage: torchx run <run args...> spmd  [--help] [--script SCRIPT] [-m M] [--image IMAGE] [--name NAME] [-h H]
                                      [-j J] [--env ENV] [--max_retries MAX_RETRIES] [--mounts MOUNTS]
                                      [--debug DEBUG]
                                      ...

    Usage (by script): torchx run spmd -j 2x8 -h p4d.24xlarge --name my_experiment/trial_1 --script
    path/to/my/trainer.py -foo bar ...

    positional arguments:
      args                  the arguments to the main module or script (e.g. my/trainer.py -foo bar) (for docker based
                            runs) the script path must be relative to the WORKDIR of the image (required)

    optional arguments:
      --help                show this help message and exit
      --script SCRIPT
      -m M, --m M           the main module name (e.g. my.module.trainer). When this option is used, the `script_args`
                            are passed as the arguments to the main module). Invoking my module is useful when the
                            relative/absolute path of the main script is unknown w.r.t the WORKDIR of the image. Use
                            this option when it makes sense to invoke the main script via `python -m <MAIN.MODULE>`.
                            (default: None)
      --image IMAGE         the base docker image of the workspace, if workspace is disabled, then the image of the
                            job (default: 763104351884.dkr.ecr.us-west-2.amazonaws.com/pytorch-training:1.13.1-gpu-
                            py39-cu117-ubuntu20.04-ec2)
      --name NAME           {experimentname}/{runname} (defaults to `default_experiment_{USER}_{HOST}/{scriptname}`)
                            (default: None)
      -h H, --h H           the type of host to run on (e.g. p4d.24xlarge). Must be one of the registered named
                            resources
      -j J, --j J           {nnodes}x{nproc_per_node}, for gpu hosts, nproc_per_node must not exceed num gpus
                            (default: 1x1)
      --env ENV             environment varibles to be passed to the run (e.g. ENV1=v1,ENV2=v2,ENV3=v3) (default:
                            None)
      --max_retries MAX_RETRIES
                            the number of scheduler retries allowed (default: 0)
      --mounts MOUNTS       (for docker based runs only) mounts to mount into the worker environment/container (ex.
                            type=<bind/volume>,src=/host,dst=/job[,readonly]). (default: None)
      --debug DEBUG         whether to run with preset debug flags enabled (default: False)

"""
import os

from getpass import getuser
from typing import Dict, List, Optional

from torchx import specs
from torchx.components.dist import (
    _args_join,
    _noquote,
    _TORCH_DEBUG_FLAGS,
    get_role_name,
)
from torchx.components.structured_arg import StructuredJArgument, StructuredNameArgument
from torchx.util.aws.region import get_region

MINUTES = 60
EFA_DEVICE = "vpc.amazonaws.com/efa"


class DLContainer:
    """
    Provides docker container image URL to the AWS DL container images.
    https://github.com/aws/deep-learning-containers/blob/master/available_images.md
    """

    @property
    def ECR_URL(self) -> str:
        return f"763104351884.dkr.ecr.{get_region()}.amazonaws.com"

    @property
    def PYTORCH_TRAINING_1_13_1_GPU(self) -> str:
        return f"{self.ECR_URL}/pytorch-training:1.13.1-gpu-py39-cu117-ubuntu20.04-ec2"

    @property
    def PYTORCH_TRAINING_2_0_1_GPU(self) -> str:
        return f"{self.ECR_URL}/pytorch-training:2.0.1-gpu-py310-cu118-ubuntu20.04-ec2"


def default_experiment_name() -> str:
    return f"default-experiment-{getuser()}"


def is_efa_enabled(resource: specs.Resource) -> bool:
    return EFA_DEVICE in resource.devices

def spmd(
    *args: str,
    script: Optional[str] = None,
    m: Optional[str] = None,
    image: Optional[str] = None,
    name: str = "/",
    h: str = "aws_p3_2xlarge",
    j: str = "1x1",
    env: Optional[Dict[str, str]] = None,
    max_retries: int = 0,
    mounts: Optional[List[str]] = None,
    local: bool = False,
    debug: bool = False,
) -> specs.AppDef:
    """
    Usage (by script): torchx run spmd -j 2x8 -h p4d.24xlarge --name my_experiment/trial_1 --script path/to/my/trainer.py -foo bar

    Usage (by module): torchx run spmd -j 2x8 -h p4d.24xlarge --name my_experiment/trial_1 -m path.to.my.trainer -foo bar

    Usage (infer GPU count): torchx run spmd -j 2 -h p4d.24xlarge ... (same as -j 2x8)

    Creates a torchx.specs.AppDef (Job Definition) for a Single-Process-Multiple-Data (SPMD)
    style application. See: https://en.wikipedia.org/wiki/Single_program,_multiple_data.

    SPMD launches `n x m` (set via the `-j nxm` option) copies of the same program,
    where `n` is the number of nodes (hosts) and `m` is the number of processes on each node.

    If you have a distributed PyTorch script (DDP, FSDP, RPC) use this component to launch
    the distributed application. You can also use `-j 1x1` to launch a single process application
    which would be equivalent to launching with regular `python` except that your application
    can safely call `torch.distributed.init_process_group(backend)`.

    Note: For multi-node distributed runs, the hosts MUST have a network route to each other
          AND port 29500 should be open on all hosts. Please check your security group settings.


    Args:
        args: the arguments to the main module or script (e.g. my/trainer.py -foo bar)
            (for docker based runs) the script path must be relative to the WORKDIR of the image
        script:
        m: the main module name (e.g. my.module.trainer). When this option is used, the `script_args` are passed
           as the arguments to the main module). Invoking my module is useful when the relative/absolute path
           of the main script is unknown w.r.t the WORKDIR of the image. Use this option when it makes sense to
           invoke the main script via `python -m <MAIN.MODULE>`.
        image: the base docker image of the workspace, if workspace is disabled, then the image of the job
        name: {experimentname}/{runname} (defaults to `default_experiment_{USER}_{HOST}/{scriptname}`)
        h: the type of host to run on (e.g. p4d.24xlarge). Must be one of the registered named resources
        j: {nnodes}x{nproc_per_node}. For GPU hosts omitting nproc_per_node will infer it from the GPU count on the host
        env: environment varibles to be passed to the run (e.g. ENV1=v1,ENV2=v2,ENV3=v3)
        max_retries: the number of scheduler retries allowed
        rdzv_port: the port on rank0's host to use for hosting the c10d store used for rendezvous.
                   Only takes effect when running multi-node. When running single node, this parameter
                   is ignored and a random free port is chosen.
        mounts: (for docker based runs only) mounts to mount into the worker environment/container
                (ex. type=<bind/volume>,src=/host,dst=/job[,readonly]).
        local: when set to `True` makes it possible to run with `nnodes > 1` on a multi-gpu host
        debug: whether to run with preset debug flags enabled

    """

    if image is None:
        image = DLContainer().PYTORCH_TRAINING_2_0_1_GPU

    resource = specs.get_named_resources(h)

    if env is None:
        env = {}

    if is_efa_enabled(resource):
        # DL container sets NCCL_SOCKET_IFNAME=^docker0 in /etc/nccl.conf
        # so we need to override it here so that NCCL properly establishes sockets
        # select eth or ens network interfaces (ens is used in modern OS like ubuntu 20)
        env["NCCL_SOCKET_IFNAME"] = "eth,ens"
        env["FI_PROVIDER"] = "efa"
        env["FI_EFA_USE_DEVICE_RDMA"] = "1"

    structured_name = StructuredNameArgument.parse_from(
        name=name,
        m=m,
        script=script,
        default_experiment_name=default_experiment_name(),
    )

    structured_j = StructuredJArgument.parse_from(h, j)

    rdzv_backend = "c10d"
    if structured_j.nnodes == 1:
        # using port 0 makes elastic chose a free random port which is ok
        # for single-node jobs since all workers run under a single agent
        # When nnodes is 0 and max_nnodes is 1, it's stil a single node job
        # but pending until the resources become available

        rdzv_endpoint = "localhost:0"
    else:
        # for multi-node, rely on the rank0_env environment variable set by
        # the schedulers (see scheduler implementation for the actual env var this maps to)
        # some schedulers (e.g. aws batch) make the rank0's ip-addr available on all BUT on rank0
        # so default to "localhost" if the env var is not set or is empty
        # rdzv_endpoint bash resolves to something to the effect of
        # ${TORCHX_RANK0_HOST:=localhost}:29500
        # use $$ in the prefix to escape the '$' literal (rather than a string Template substitution argument)
        rdzv_endpoint = _noquote(f"$${{{specs.macros.rank0_env}:=localhost}}:29500")

    env.setdefault("LOGLEVEL", os.getenv("LOGLEVEL", "INFO"))
    if debug:
        env.update(_TORCH_DEBUG_FLAGS)
    env["TORCHX_TRACKING_EXPERIMENT_NAME"] = structured_name.experiment_name
    env["TORCHX_USER"] = getuser()
    # Increase timeout and retry values for requests made to Instance Metadata server.
    env["AWS_METADATA_SERVICE_TIMEOUT"] = "5"  # seconds
    env["AWS_METADATA_SERVICE_NUM_ATTEMPTS"] = "10"
    env["APE_COMPONENT_NAME"] = "spmd"
    env["APE_NNODES"] = str(structured_j.nnodes)
    env["APE_NPROC_PER_NODE"] = str(structured_j.nproc_per_node)

    cmd = [
        "torchrun",
        "--rdzv_backend",
        rdzv_backend,
        "--rdzv_endpoint",
        rdzv_endpoint,
        "--rdzv_id",
        f"{specs.macros.app_id}",
        "--rdzv_conf",
        # configure various rdzv timeouts
        # see: https://github.com/pytorch/pytorch/blob/main/torch/distributed/elastic/rendezvous/dynamic_rendezvous.py#L1216
        # join_timeout == wait 30 minutes for other nodes to join the job
        # close_timeout == wait 10 minutes for all nodes to close rdzv
        # timeout == 30min (catch-all) not really used for c10d rdzv but having it here in case we switch to another backend
        f"join_timeout={30 * MINUTES},close_timeout={10 * MINUTES},timeout={30*MINUTES}",
        "--nnodes",
        str(structured_j.nnodes),
        "--nproc_per_node",
        str(structured_j.nproc_per_node),
        "--tee",
        "3",
        "--role",
        "",
    ]
    if script is not None:
        cmd += [script]
    else:  # m is not None (this is checked in StructuredNameArgument.parse_from() call above)
        cmd += ["-m", m]
    cmd += args
    resource = specs.resource(h=h)

    if local:
        # override num gpus to be equal to the specified nproc_per_node
        # this makes torchx's LocalScheduler be able to correctly set CUDA_VISIBLE_DEVICES
        # for each simulated node
        resource.gpu = structured_j.nproc_per_node

    return specs.AppDef(
        name=structured_name.run_name,
        roles=[
            specs.Role(
                name=get_role_name(script, m),
                image=image,
                min_replicas=structured_j.nnodes,
                entrypoint="bash",
                num_replicas=structured_j.nnodes,
                resource=resource,
                args=["-c", _args_join(cmd)],
                env=env,
                port_map={
                    "c10d": 29500,
                },
                max_retries=max_retries,
                mounts=specs.parse_mounts(mounts) if mounts else [],
            )
        ],
    )
