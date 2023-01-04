import torchx.components.hpo as hpo
from torchx.components.component_test_base import ComponentTestCase


def _to_dict(iterable):
    # Assumes odd elements are the key and even elements are values
    mapping = {}
    for i, el in enumerate(iterable):
        if i % 2 == 0:
            key = el
        else:
            mapping[key] = el
    return mapping


class HpoTestCase(ComponentTestCase):

    def setUp(self):
        super().setUp()
        self.args = {
            "eval_fn": "torchx.apps.utils.booth:main",
            "objective": "accuracy",
            "hpo_params_file": "parameters.json",
            "hpo_trials": 100,
            "hpo_maximize": True,
            "name": "appname",
            "cpu": 16,
            "gpu": 2,
            "memMB": 1024,
            "image": "image",
            "env": {
                "VAR": "VAL"
            },
            "max_retries": 2,
            "mounts": ["type=bind", "src=/dst", "dst=/dst", "readonly"],
        }

    def test_grid_search(self) -> None:
        self.validate(hpo, "grid_search")

    def test_bayesian(self) -> None:
        self.validate(hpo, "bayesian")

    def test_grid_search_arguments(self) -> None:
        app = hpo.grid_search(**self.args)

        self.assertEqual(app.name, self.args["name"])
        self.assertEqual(1, len(app.roles))
        role = app.roles[0]
        self.assertEqual(role.name, self.args["eval_fn"])
        self.assertEqual(role.image, self.args["image"])
        for k, v in self.args["env"].items():
            self.assertEqual(role.env[k], v)
        self.assertEqual(role.max_retries, self.args["max_retries"])
        self.assertEqual(len(role.mounts), 1)
        self.assertEqual(role.resource.cpu, self.args["cpu"])
        self.assertEqual(role.resource.gpu, self.args["gpu"])
        self.assertEqual(role.resource.memMB, self.args["memMB"])

        entrypoint = role.entrypoint

        actual_args = _to_dict(role.args)
        self.assertEqual(actual_args["-m"], "torchx.components.hpo_runner")
        self.assertEqual(actual_args["--eval_fn"],
                         "torchx.apps.utils.booth:main")
        self.assertEqual(actual_args["--objective"], self.args["objective"])
        self.assertEqual(actual_args["--hpo_params_file"],
                         self.args["hpo_params_file"])
        self.assertEqual(actual_args["--hpo_strategy"], "grid_search")
        self.assertEqual(actual_args["--hpo_trials"],
                         str(self.args["hpo_trials"]))

    def test_bayesian_arguments(self) -> None:
        app = hpo.bayesian(**self.args)

        self.assertEqual(app.name, self.args["name"])
        self.assertEqual(1, len(app.roles))
        role = app.roles[0]
        self.assertEqual(role.name, self.args["eval_fn"])
        self.assertEqual(role.image, self.args["image"])
        for k, v in self.args["env"].items():
            self.assertEqual(role.env[k], v)
        self.assertEqual(role.max_retries, self.args["max_retries"])
        self.assertEqual(len(role.mounts), 1)
        self.assertEqual(role.resource.cpu, self.args["cpu"])
        self.assertEqual(role.resource.gpu, self.args["gpu"])
        self.assertEqual(role.resource.memMB, self.args["memMB"])

        entrypoint = role.entrypoint

        actual_args = _to_dict(role.args)
        self.assertEqual(actual_args["-m"], "torchx.components.hpo_runner")
        self.assertEqual(actual_args["--eval_fn"],
                         "torchx.apps.utils.booth:main")
        self.assertEqual(actual_args["--objective"], self.args["objective"])
        self.assertEqual(actual_args["--hpo_params_file"], self.args["hpo_params_file"])
        self.assertEqual(actual_args["--hpo_strategy"], "bayesian")
        self.assertEqual(actual_args["--hpo_trials"], str(self.args["hpo_trials"]))
