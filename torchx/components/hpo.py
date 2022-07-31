# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import os
from typing import Dict, List, Optional

import torchx
import torchx.specs as specs


def _build_hpo_app(
    eval_fn: str,
    objective: str,
    hpo_params_file: str,
    hpo_strategy: str,
    hpo_trials: int,
    hpo_maximize: bool,
    name: Optional[str],
    cpu: int,
    gpu: int,
    memMB: int,
    h: Optional[str],
    image: str,
    env: Optional[Dict[str, str]],
    max_retries: int,
    mounts: Optional[List[str]],
) -> specs.AppDef:
    role_name = eval_fn
    if env is None:
        env = {}
    env.setdefault("LOGLEVEL", os.getenv("LOGLEVEL", "WARNING"))

    opt_args = []
    if hpo_trials:
        opt_args.extend(["--hpo_trials", str(hpo_trials)])
    if hpo_maximize:
        opt_args.extend(["--hpo_maximize"])
    return specs.AppDef(
        name=name or role_name,
        roles=[
            specs.Role(
                name=role_name,
                image=image,
                entrypoint="python",
                args=[
                    *[
                        "-m",
                        "torchx.components.hpo_runner",
                        "--eval_fn",
                        f"{eval_fn}",
                        "--objective",
                        f"{objective}",
                        "--hpo_params_file",
                        f"{hpo_params_file}",
                        "--hpo_strategy",
                        f"{hpo_strategy}",
                    ],
                    *opt_args,
                ],
                resource=specs.resource(cpu=cpu, gpu=gpu, memMB=memMB, h=h),
                env=env,
                max_retries=max_retries,
                mounts=specs.parse_mounts(mounts) if mounts else [],
            )
        ],
    )


def bayesian(
    eval_fn: str,
    objective: str = None,
    hpo_params_file: str = None,
    hpo_trials: int = None,
    hpo_maximize: bool = False,
    name: Optional[str] = None,
    cpu: int = 1,
    gpu: int = 0,
    memMB: int = 1024,
    h: Optional[str] = None,
    image: str = torchx.IMAGE,
    env: Optional[Dict[str, str]] = None,
    max_retries: int = 0,
    mounts: Optional[List[str]] = None,
) -> specs.AppDef:
    """
        Hyper-parameter optimization application that uses bayesian candidate selection approach.

        Notes: Utilizes `torchx.components.hpo_runner` module to control the experiment definition and
        execution, which currently:
        - Uses Ax library for candidate selection
        - Limited to sequential single-arm processing
        - Bayesian candidate selection will run initial grid-search like exploration for initial BO input
        For now results are printed out, with a plan to migrate to proper tracking solution.

        Args:
            eval_fn: Object reference entry-point, eg. `mymodule:my_training_function`. Input is a map with
                     properties defined in `hpo_params_file` and output is a map with results.
            objective: property that is in output map and will be used as a performance metric
            hpo_prams_file: Parameter json file in following format {"params": {"PARAM_NAME_1": {"type": "[float|int|str|bool]", "[range|choice]":["v1", "v2"] }}}
            hpo_trials: number of trials to execute
            hpo_maximize: expected objective value direction
            name: job name override (uses the eval_fn as name if not specified)
            cpu: number of cpus per replica
            gpu: number of gpus per replica
            memMB: cpu memory in MB per replica
            h: a registered named resource (if specified takes precedence over cpu, gpu, memMB)
            image: image (e.g. docker)
            env: environment properties (K/Vs)
            max_retries: the number of scheduler retries allowed
            mounts: mounts to mount into the worker environment/container (ex. type=<bind/volume>,src=/host,dst=/job[,readonly]).
                    See scheduler documentation for more info.

    """
    return _build_hpo_app(
        eval_fn=eval_fn,
        objective=objective,
        hpo_params_file=hpo_params_file,
        hpo_strategy="bayesian",
        hpo_trials=hpo_trials,
        hpo_maximize=hpo_maximize,
        name=name,
        cpu=cpu,
        gpu=gpu,
        memMB=memMB,
        h=h,
        image=image,
        env=env,
        max_retries=max_retries,
        mounts=mounts,
    )


def grid_search(
    eval_fn: str,
    objective: str = None,
    hpo_params_file: str = None,
    hpo_trials: int = None,
    hpo_maximize: bool = False,
    name: Optional[str] = None,
    cpu: int = 1,
    gpu: int = 0,
    memMB: int = 1024,
    h: Optional[str] = None,
    image: str = torchx.IMAGE,
    env: Optional[Dict[str, str]] = None,
    max_retries: int = 0,
    mounts: Optional[List[str]] = None,
) -> specs.AppDef:
    """
    Hyper-parameter optimization application that uses grid search candidate selection approach.

    Notes: Utilizes `torchx.components.hpo_runner` module to control the experiment definition and execution, which currently:
    - Uses Ax library for candidate selection
    - Limited to sequential single-arm processing
    - Instead of uniform grid search, uses SOBOL strategy
    For now results are printed out, with a plan to migrate to proper tracking solution.

    Args:
    Args:
        eval_fn: Object reference entry-point, eg. `mymodule:my_training_function`. Input is a map with
                 properties defined in `hpo_params_file` and output is a map with results.
        objective: property that is in output map and will be used as a performance metric
        hpo_prams_file: Parameter json file in following format {"params": {"PARAM_NAME_1": {"type": "[float|int|str|bool]", "[range|choice]":["v1", "v2"] }}}
        hpo_trials: number of trials to execute
        hpo_maximize: expected objective value direction
        name: job name override (uses the eval_fn as name if not specified)
        cpu: number of cpus per replica
        gpu: number of gpus per replica
        memMB: cpu memory in MB per replica
        h: a registered named resource (if specified takes precedence over cpu, gpu, memMB)
        image: image (e.g. docker)
        env: environment properties (K/Vs)
        max_retries: the number of scheduler retries allowed
        mounts: mounts to mount into the worker environment/container (ex. type=<bind/volume>,src=/host,dst=/job[,readonly]).
                See scheduler documentation for more info.
    """

    return _build_hpo_app(
        eval_fn=eval_fn,
        objective=objective,
        hpo_params_file=hpo_params_file,
        hpo_strategy="grid_search",
        hpo_trials=hpo_trials,
        hpo_maximize=hpo_maximize,
        name=name,
        cpu=cpu,
        gpu=gpu,
        memMB=memMB,
        h=h,
        image=image,
        env=env,
        max_retries=max_retries,
        mounts=mounts,
    )
