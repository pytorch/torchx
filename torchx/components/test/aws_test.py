
import shlex
import unittest

from torchx.components.aws import default_experiment_name, spmd, MINUTES
from torchx.components.dist import _TORCH_DEBUG_FLAGS
from torchx.specs import macros


class SpmdTest(unittest.TestCase):
    def test_spmd_call_by_module_or_script_no_name(self) -> None:
        appdef = spmd(script="foo/bar.py")
        self.assertEqual("bar", appdef.name)
        self.assertEqual(
            default_experiment_name(),
            appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"],
        )

        appdef = spmd("-a", "b", script="foo/bar.py")
        self.assertEqual("bar", appdef.name)
        self.assertEqual(
            default_experiment_name(),
            appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"],
        )

        appdef = spmd(m="foo.bar")
        self.assertEqual("bar", appdef.name)
        self.assertEqual(
            default_experiment_name(),
            appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"],
        )

        appdef = spmd("-a", "b", m="foo.bar")
        self.assertEqual("bar", appdef.name)
        self.assertEqual(
            default_experiment_name(),
            appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"],
        )

        with self.assertRaises(ValueError):
            spmd()

        with self.assertRaises(ValueError):
            spmd(m="foo.bar", script="foo/bar.py")

    def test_spmd_call_by_module_or_script_with_name(self) -> None:
        appdef = spmd(script="foo/bar.py", name="baz/trial_1")
        self.assertEqual("trial_1", appdef.name)
        self.assertEqual("baz", appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"])

        appdef = spmd("-a", "b", script="foo/bar.py", name="baz/trial_1")
        self.assertEqual("trial_1", appdef.name)
        self.assertEqual("baz", appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"])

        appdef = spmd(m="foo.bar", name="baz/trial_1")
        self.assertEqual("trial_1", appdef.name)
        self.assertEqual("baz", appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"])

        appdef = spmd("-a", "b", m="foo.bar", name="baz/trial_1")
        self.assertEqual("trial_1", appdef.name)
        self.assertEqual("baz", appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"])

    def test_spmd_call_by_module_or_script_with_experiment_name(self) -> None:
        appdef = spmd(script="foo/bar.py", name="baz/")
        self.assertEqual("bar", appdef.name)
        self.assertEqual("baz", appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"])

        appdef = spmd("-a", "b", script="foo/bar.py", name="baz/")
        self.assertEqual("bar", appdef.name)
        self.assertEqual("baz", appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"])

        appdef = spmd(m="foo.bar", name="baz/")
        self.assertEqual("bar", appdef.name)
        self.assertEqual("baz", appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"])

        appdef = spmd("-a", "b", m="foo.bar", name="baz/")
        self.assertEqual("bar", appdef.name)
        self.assertEqual("baz", appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"])

    def test_spmd_call_by_module_or_script_with_run_name(self) -> None:
        appdef = spmd(script="foo/bar.py", name="/trial_1")
        self.assertEqual("trial_1", appdef.name)
        self.assertEqual(
            default_experiment_name(),
            appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"],
        )

        appdef = spmd("-a", "b", script="foo/bar.py", name="/trial_1")
        self.assertEqual("trial_1", appdef.name)
        self.assertEqual(
            default_experiment_name(),
            appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"],
        )

        appdef = spmd(m="foo.bar", name="/trial_1")
        self.assertEqual("trial_1", appdef.name)
        self.assertEqual(
            default_experiment_name(),
            appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"],
        )

        appdef = spmd("-a", "b", m="foo.bar", name="/trial_1")
        self.assertEqual("trial_1", appdef.name)
        self.assertEqual(
            default_experiment_name(),
            appdef.roles[0].env["TORCHX_TRACKING_EXPERIMENT_NAME"],
        )

    def test_spmd_single_node(self) -> None:
        appdef = spmd(script="foo/bar.py", j="1x4", h="p3.8xlarge")
        role = appdef.roles[0]
        args = role.args
        self.assertEqual("-c", args[0])

        cmd = shlex.split(args[1])
        self.assertEqual("bash", role.entrypoint)
        self.assertEqual("c10d", cmd[cmd.index("--rdzv_backend") + 1])
        self.assertEqual("localhost:0", cmd[cmd.index("--rdzv_endpoint") + 1])
        self.assertEqual(
            f"join_timeout={30*MINUTES},close_timeout={10*MINUTES},timeout={30*MINUTES}",
            cmd[cmd.index("--rdzv_conf") + 1],
        )
        self.assertEqual("1", cmd[cmd.index("--nnodes") + 1])
        self.assertEqual("4", cmd[cmd.index("--nproc_per_node") + 1])
        self.assertTrue("--node_rank" not in cmd)

    def test_spmd_multi_node(self) -> None:
        appdef = spmd(script="foo/bar.py", j="2x4", h="p3.8xlarge")
        role = appdef.roles[0]
        args = role.args

        self.assertEqual("-c", args[0])

        cmd = shlex.split(args[1])
        self.assertEqual("bash", role.entrypoint)
        self.assertEqual("c10d", cmd[cmd.index("--rdzv_backend") + 1])
        self.assertTrue(
            f"--rdzv_endpoint $${{{macros.rank0_env}:=localhost}}:29500" in args[1]
        )
        self.assertEqual("2", cmd[cmd.index("--nnodes") + 1])
        self.assertEqual("4", cmd[cmd.index("--nproc_per_node") + 1])

        # by default, we assign 1 gpu per process
        self.assertEqual(4, role.resource.gpu)

    def test_spmd_multi_node_multi_gpu_per_proc(self) -> None:
        appdef = spmd(script="foo/bar.py", j="2x4", h="p3.16xlarge")
        role = appdef.roles[0]
        args = role.args

        self.assertEqual("-c", args[0])

        cmd = shlex.split(args[1])
        self.assertEqual("bash", role.entrypoint)
        self.assertEqual("c10d", cmd[cmd.index("--rdzv_backend") + 1])
        self.assertTrue(
            f"--rdzv_endpoint $${{{macros.rank0_env}:=localhost}}:29500" in args[1]
        )
        self.assertEqual("2", cmd[cmd.index("--nnodes") + 1])
        self.assertEqual("4", cmd[cmd.index("--nproc_per_node") + 1])

        # intentional mismatch of the number of GPUs in the node and the number of processes
        self.assertEqual(8, role.resource.gpu)

    def test_spmd_efa_enabled(self) -> None:
        appdef = spmd(script="foo/bar.py", h="p4d.24xlarge")
        env = appdef.roles[0].env
        self.assertEqual("eth,ens", env["NCCL_SOCKET_IFNAME"])
        self.assertEqual("1", env["FI_EFA_USE_DEVICE_RDMA"])

    def test_instance_metadata_server_config(self) -> None:
        appdef = spmd(script="foo/bar.py", h="p4d.24xlarge")
        env = appdef.roles[0].env
        self.assertEqual("5", env["AWS_METADATA_SERVICE_TIMEOUT"])
        self.assertEqual("10", env["AWS_METADATA_SERVICE_NUM_ATTEMPTS"])

    def test_spmd_debug_mode(self) -> None:
        appdef = spmd(script="foo/bar.py", debug=True)
        env = appdef.roles[0].env

        for k, v in _TORCH_DEBUG_FLAGS.items():
            self.assertEqual(v, env[k])

    def test_spmd_custom_env(self) -> None:
        appdef = spmd(script="foo/bar.py", env={"FOO": "BAR"})
        self.assertEqual("BAR", appdef.roles[0].env["FOO"])

    def test_spmd_no_main(self) -> None:
        with self.assertRaisesRegex(ValueError, "No main module or script specified."):
            spmd()
