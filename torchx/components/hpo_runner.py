from dataclasses import dataclass
import sys
import importlib
import json
from enum import Enum
from ax.service.ax_client import AxClient
from ax.modelbridge.registry import Models
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep

from typing import (
    Dict,
    List,
    TypedDict,
    Union,
)

import argparse


def _arg_parser():
    arg_parser = argparse.ArgumentParser(description="HPO Runner args")
    arg_parser.add_argument(
        "--eval_fn",
        type=str,
        help="Fully qualified run function name. eg. my_modue:train_func",
    )
    arg_parser.add_argument("--objective",
                            type=str,
                            help="objective key in eval_fn")
    arg_parser.add_argument("--hpo_params_file",
                            type=str,
                            help="HPO parameters")
    arg_parser.add_argument(
        "--hpo_strategy",
        type=str,
        choices=["grid_search", "bayesian", "auto"],
        default="auto",
        help="HPO selection strategy. Auto - may generate hybrid strategy",
    )
    arg_parser.add_argument(
        "--hpo_trials",
        type=int,
        default=20,
        help="Maximun number of trials to run.",
    )
    arg_parser.add_argument(
        "--hpo_maximize",
        default=False,
        action="store_true",
        help="Optimization direction. Default action is to minimize",
    )

    return arg_parser


class SearchSpaceParamType(str, Enum):
    INT = "int"
    FLOAT = "float"
    STRING = "str"
    BOOL = "bool"


class RangeConstraint(TypedDict):
    upper: Union[int, float]
    lower: Union[int, float]


@dataclass
class SearchSpaceParamRangeConstraint:
    range: RangeConstraint


@dataclass
class SearchSpaceParamChoiceConstraint:
    choices: List[Union[int, float, str, bool]]


SearchSpaceParamConstraint = Union[SearchSpaceParamRangeConstraint,
                                   SearchSpaceParamChoiceConstraint]


@dataclass
class SearchSpaceParam:
    name: str
    param_type: SearchSpaceParamType
    constraints: List[SearchSpaceParamConstraint]


def _parse_param(name, json_data) -> SearchSpaceParam:
    constraint = None
    cast_op = str
    if json_data["type"] == SearchSpaceParamType.INT:
        cast_op = int
    elif json_data["type"] == SearchSpaceParamType.FLOAT:
        cast_op = float

    if "range" in json_data:
        assert len(
            json_data["range"]) == 2, "Range requires start and end values"
        lower = cast_op(json_data["range"][0])
        upper = cast_op(json_data["range"][1])
        if upper < lower:
            lower, upper = upper, lower
        constraint = RangeConstraint(upper=upper, lower=lower)
    elif "choice" in json_data:
        choices = [cast_op(v) for v in json_data["choice"]]
        constraint = SearchSpaceParamChoiceConstraint(choices)
    else:
        raise ValueError(
            "Parameters must have either 'range' or 'choice' constraints defined"
        )
    return SearchSpaceParam(name=name,
                            param_type=json_data["type"],
                            constraints=[constraint])


def _parse_params(json_data) -> List[SearchSpaceParam]:
    if "params" in json_data:
        return [
            _parse_param(name, props)
            for name, props in json_data["params"].items()
        ]
    else:
        return []


class AxClientBackedHPO:

    def sobol_gs_factory(self, hpo_trials):
        return GenerationStrategy(steps=[
            GenerationStep(
                model=Models.SOBOL,
                num_trials=hpo_trials,
                max_parallelism=5,
            ),
        ])

    def sobol_gpei_gs_factory(self, initial_trials, gpei_trials):
        return GenerationStrategy(steps=[
            GenerationStep(
                model=Models.SOBOL,
                num_trials=initial_trials,
                min_trials_observed=min(initial_trials, 3),
                max_parallelism=5,
            ),
            GenerationStep(
                model=Models.GPEI,
                num_trials=gpei_trials,
                max_parallelism=3,
            ),
        ])

    def __init__(self, ax_client=None):
        # allow injection for testing purposes
        self.ax_client = ax_client

    def _params_to_ax_client(self, params) -> List[Dict]:
        ax_client_params = []
        for param in params:
            ax_client_param = {}
            ax_client_param["name"] = param.name

            if type(param.constraints[0]) == SearchSpaceParamChoiceConstraint:
                ax_client_param["type"] = "choice"
                ax_client_param["values"] = param.constraints[0].choices
            else:
                ax_client_param["type"] = "range"
                ax_client_param["bounds"] = [
                    param.constraints[0]["lower"],
                    param.constraints[0]["upper"],
                ]
            ax_client_params.append(ax_client_param)

        return ax_client_params

    def _optimize(
        self,
        params: List[SearchSpaceParam],
        eval_callable,
        objective,
        minimize,
        hpo_trials,
        generation_strategy: GenerationStrategy = None,
    ) -> None:
        if not self.ax_client:
            self.ax_client = AxClient(generation_strategy=generation_strategy)
        ax_params = self._params_to_ax_client(params)

        self.ax_client.create_experiment(
            name="torchx_hpo_experiment",
            parameters=ax_params,
            objective_name=objective,
            minimize=minimize,
        )

        for i in range(hpo_trials):
            parameters, trial_index = self.ax_client.get_next_trial()
            self.ax_client.complete_trial(trial_index=trial_index,
                                          raw_data=eval_callable(parameters))

        best_parameters, values = self.ax_client.get_best_parameters()
        return best_parameters

    def grid_search(
        self,
        params: List[SearchSpaceParam],
        eval_callable,
        objective,
        minimize,
        hpo_trials,
    ) -> None:

        return self._optimize(
            params,
            eval_callable,
            objective,
            minimize,
            hpo_trials,
            generation_strategy=self.sobol_gs_factory(hpo_trials),
        )

    def bayesian(
        self,
        params: List[SearchSpaceParam],
        eval_callable,
        objective,
        minimize,
        hpo_trials,
    ) -> None:
        if hpo_trials < 2:
            raise ValueError(
                "Bayesian optimization requires more than one trial")
        initial_trials = 5
        if initial_trials + 1 > hpo_trials:
            initial_trials = 1

        return self._optimize(
            params,
            eval_callable,
            objective,
            minimize,
            hpo_trials,
            generation_strategy=self.sobol_gpei_gs_factory(
                initial_trials, hpo_trials - initial_trials),
        )

    def auto(
        self,
        params: List[SearchSpaceParam],
        eval_callable,
        objective,
        minimize,
        hpo_trials,
    ) -> None:

        return self._optimize(params, eval_callable, objective, minimize,
                              hpo_trials)


def _run(args):
    modname, qualname_separator, qualname = args.eval_fn.partition(":")
    mod = importlib.import_module(modname)
    eval_fn_ref = getattr(mod, qualname)

    params = None
    with open(args.hpo_params_file) as f:
        params_json = json.load(f)
        params = _parse_params(params_json)

    minimize = False if args.hpo_maximize else True
    ax_client_api_based_hpo = AxClientBackedHPO()
    if args.hpo_strategy == "grid_search":
        candidate_optimal_properties = ax_client_api_based_hpo.grid_search(
            params, eval_fn_ref, args.objective, minimize, args.hpo_trials)
    elif args.hpo_strategy == "bayesian":
        candidate_optimal_properties = ax_client_api_based_hpo.bayesian(
            params, eval_fn_ref, args.objective, minimize, args.hpo_trials)
    else:
        candidate_optimal_properties = ax_client_api_based_hpo.auto(
            params, eval_fn_ref, args.objective, minimize, args.hpo_trials)

    return candidate_optimal_properties


def main():
    arg_parser = _arg_parser()
    args = arg_parser.parse_args(sys.argv[1:])
    _run(args)


if __name__ == "__main__":
    main()
