import torchx.components.hpo_runner as hpo_runner
from torchx.components.hpo_runner import (
    SearchSpaceParamType,
    SearchSpaceParam,
    SearchSpaceParamChoiceConstraint,
)

import unittest
from unittest.mock import Mock, MagicMock, patch, mock_open
import json


def booth_fn(params):
    x1 = params['x1']
    x2 = params['x2']
    val = (x1 + 2 * x2 - 7)**2 + (2 * x1 + x2 - 5)**2
    return {"val": val}


class HpoRunnerTestCase(unittest.TestCase):

    def setUp(self):
        self.args = [
            "--eval_fn",
            "torchx.components.test.hpo_runner_test:booth_fn",
            "--objective",
            "val",
            "--hpo_params_file",
            "hpo_params_file.json",
            "--hpo_strategy",
            "auto",
        ]
        self.params_json = '{"params": \
{ \
 "x1": { \
  "type": "float", \
  "range": ["0.01", "0.1"] }, \
"x2": { \
  "type": "int", \
  "choice" : [1, 2, 3] \
 } \
} \
}'

    def test_parse_params(self):

        params = hpo_runner._parse_params(json.loads(self.params_json))
        self.assertEqual(2, len(params))
        self.assertEqual("x1", params[0].name)

        self.assertEqual(SearchSpaceParamType.FLOAT, params[0].param_type)
        self.assertEqual(0.01, params[0].constraints[0]["lower"])
        self.assertEqual(0.1, params[0].constraints[0]["upper"])

        self.assertEqual("x2", params[1].name)

        self.assertEqual(SearchSpaceParamType.INT, params[1].param_type)

        self.assertEqual([1, 2, 3], params[1].constraints[0].choices)

    def test_arg_parser(self):
        arg_parser = hpo_runner._arg_parser()
        parsed_args = arg_parser.parse_args(self.args)
        self.assertEqual(parsed_args.eval_fn,
                         "torchx.components.test.hpo_runner_test:booth_fn")
        self.assertEqual(parsed_args.objective, "val")
        self.assertEqual(parsed_args.hpo_params_file, "hpo_params_file.json")
        self.assertEqual(parsed_args.hpo_strategy, "auto")

    def test_run_auto(self):
        parsed_args = hpo_runner._arg_parser().parse_args(self.args)
        with patch("builtins.open",
                   mock_open(read_data=self.params_json)) as mock_file:
            with patch("torchx.components.hpo_runner.AxClientBackedHPO.auto"
                       ) as auto_strategy_mock:
                result = {"k": "v"}
                auto_strategy_mock.return_value = result
                hpo_runner._run(parsed_args)
                auto_strategy_mock.assert_called_once()

    def test_run_grid_search(self):
        parsed_args = hpo_runner._arg_parser().parse_args(
            self.args + ["--hpo_strategy", "grid_search"])
        with patch("builtins.open",
                   mock_open(read_data=self.params_json)) as mock_file:
            with patch(
                    "torchx.components.hpo_runner.AxClientBackedHPO.grid_search"
            ) as grid_search_mock:
                result = {"k": "v"}
                grid_search_mock.return_value = result
                hpo_runner._run(parsed_args)
                grid_search_mock.assert_called_once()

    def test_run_bayesian(self):
        parsed_args = hpo_runner._arg_parser().parse_args(
            self.args + ["--hpo_strategy", "bayesian"])
        with patch("builtins.open",
                   mock_open(read_data=self.params_json)) as mock_file:
            with patch(
                    "torchx.components.hpo_runner.AxClientBackedHPO.bayesian"
            ) as bayesian_mock:
                result = {"k": "v"}
                bayesian_mock.return_value = result
                actual_result = hpo_runner._run(parsed_args)
                bayesian_mock.assert_called_once()


class AxClientBackedHPOTest(unittest.TestCase):

    def setUp(self):
        self.ax_client_mock = Mock()
        self.optimizer = hpo_runner.AxClientBackedHPO(self.ax_client_mock)
        self.param_space = [
            SearchSpaceParam(
                name="x1",
                param_type="float",
                constraints=[{
                    "upper": 0.1,
                    "lower": 0.01
                }],
            ),
            SearchSpaceParam(
                name="x2",
                param_type="int",
                constraints=[
                    SearchSpaceParamChoiceConstraint(choices=[1, 2, 3])
                ],
            ),
        ]

    def test_to_ax_params(self):
        output = self.optimizer._params_to_ax_client(self.param_space)

        self.assertEqual("x1", output[0]["name"])
        self.assertEqual("range", output[0]["type"])
        self.assertEqual([0.01, 0.1], output[0]["bounds"])
        self.assertEqual("x2", output[1]["name"])
        self.assertEqual("choice", output[1]["type"])
        self.assertEqual([1, 2, 3], output[1]["values"])

    def test_optimize_with_auto_strategy(self):
        expected_best_params = {"x1": 1, "x2": 2}
        hpo_trials = 10

        self.ax_client_mock.get_next_trial = MagicMock(return_value=[{
            "x1": 1,
            "x2": 2
        }, 0])

        self.ax_client_mock.get_best_parameters = MagicMock(
            return_value=[expected_best_params, 10])

        actual_best_params = self.optimizer.auto(
            self.param_space,
            lambda px: {"result": px["x1"] + px["x2"]},
            "result",
            True,
            hpo_trials,
        )

        self.ax_client_mock.create_experiment.assert_called_with(
            name="torchx_hpo_experiment",
            parameters=self.optimizer._params_to_ax_client(self.param_space),
            objective_name="result",
            minimize=True,
        )

        self.assertEqual(actual_best_params, expected_best_params)

    def test_optimize_with_grid_search_strategy(self):
        hpo_trials = 10
        expected_best_params = {"x1": 1, "x2": 2}
        self.ax_client_mock.get_next_trial = MagicMock(return_value=[{
            "x1": 1,
            "x2": 2
        }, 0])

        self.ax_client_mock.get_best_parameters = MagicMock(
            return_value=[expected_best_params, 10])

        actual_best_params = self.optimizer.grid_search(
            self.param_space,
            lambda px: {"result": px["x1"] + px["x2"]},
            "result",
            True,
            hpo_trials,
        )

        self.assertEqual(actual_best_params, expected_best_params)

    def test_optimize_with_bayesian_strategy(self):
        hpo_trials = 10
        expected_best_params = {"x1": 1, "x2": 2}
        self.ax_client_mock.get_next_trial = MagicMock(return_value=[{
            "x1": 1,
            "x2": 2
        }, 0])

        self.ax_client_mock.get_best_parameters = MagicMock(
            return_value=[expected_best_params, 10])

        actual_best_params = self.optimizer.bayesian(
            self.param_space,
            lambda px: {"result": px["x1"] + px["x2"]},
            "result",
            True,
            hpo_trials,
        )

        self.assertEqual(actual_best_params, expected_best_params)
