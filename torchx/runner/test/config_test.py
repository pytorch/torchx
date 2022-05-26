#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import tempfile
import unittest
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional
from unittest.mock import patch

from torchx.runner.config import (
    apply,
    dump,
    get_config,
    get_configs,
    load,
    load_sections,
)
from torchx.schedulers import get_schedulers, Scheduler
from torchx.schedulers.api import DescribeAppResponse, Stream
from torchx.specs import AppDef, AppDryRunInfo, CfgVal, runopts


class TestScheduler(Scheduler):
    def __init__(self) -> None:
        super().__init__("_", "_")

    def schedule(self, dryrun_info: AppDryRunInfo) -> str:
        raise NotImplementedError()

    def _submit_dryrun(self, app: AppDef, cfg: Mapping[str, CfgVal]) -> AppDryRunInfo:
        raise NotImplementedError()

    def describe(self, app_id: str) -> Optional[DescribeAppResponse]:
        raise NotImplementedError()

    def _cancel_existing(self, app_id: str) -> None:
        raise NotImplementedError()

    def log_iter(
        self,
        app_id: str,
        role_name: str,
        k: int = 0,
        regex: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        should_tail: bool = False,
        streams: Optional[Stream] = None,
    ) -> Iterable[str]:
        raise NotImplementedError()

    def run_opts(self) -> runopts:
        opts = runopts()
        opts.add(
            "i",
            type_=int,
            default=1,
            help="an int option",
        )
        opts.add(
            "f",
            type_=float,
            default=1.2,
            help="a float option",
        )
        opts.add(
            "s",
            type_=str,
            default="foobar",
            help="an str option",
        )
        opts.add(
            "bTrue",
            type_=bool,
            default=True,
            help="an bool_true option",
        )
        opts.add(
            "bFalse",
            type_=bool,
            default=False,
            help="an bool_false option",
        )
        opts.add(
            "l",
            type_=List[str],
            default=["a", "b", "c"],
            help="a list option",
        )
        opts.add(
            "l_none",
            type_=List[str],
            default=None,
            help="a None list option",
        )
        opts.add(
            "empty",
            type_=str,
            default=None,
            help="an empty option",
        )
        return opts


_CONFIG = """#
[local_cwd]
log_dir = /home/bob/logs
prepend_cwd = True
"""

_CONFIG_INVALID = """#
[test]
a_run_opt_that = does_not_exist
s = option_that_exists
"""

_TEAM_CONFIG = """#
[test]
s = team_default
i = 50
f = 1.2
"""

_MY_CONFIG = """#
[test]
s = my_default
i = 100
"""

_MY_CONFIG2 = """#
[test]
s = my_default
i = 200
"""

_COMPONENT_CONFIG_0 = """#
[component:dist.ddp]
image = barbaz
script_args = a b c --d=e --f g

[component:utils.echo]
msg = abracadabra

# bad section
[component:]
foo = bar

[:bad]
leads = with delimiter

[local]
1 = 2
"""

_COMPONENT_CONFIG_1 = """#
[component:dist.ddp]
j = 1x2
image = foobar
env = A=B,C=D

[component:utils.touch]
file = tmp.txt
"""

PATH_CWD = "torchx.runner.config.Path.cwd"
TORCHX_GET_SCHEDULERS = "torchx.runner.config.get_schedulers"


class ConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = tempfile.mkdtemp(prefix="torchx_runner_config_test")
        self._write(
            ".torchxconfig",
            _TEAM_CONFIG,
        )
        self._write(
            os.path.join("home", "bob", ".torchxconfig"),
            _MY_CONFIG,
        )
        self._write(
            f"{self.test_dir}/another_torchx_config",
            _MY_CONFIG2,
        )

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir)

    def _write(self, filename: str, content: str) -> Path:
        f = Path(self.test_dir) / filename
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "w") as fp:
            fp.write(content)
        return f

    def test_load_component_defaults(self) -> None:
        configdir0 = Path(self.test_dir) / "test_load_component_defaults" / "0"
        configdir1 = Path(self.test_dir) / "test_load_component_defaults" / "1"
        configdir0.mkdir(parents=True, exist_ok=True)
        configdir1.mkdir(parents=True, exist_ok=True)
        self._write(str(configdir0 / ".torchxconfig"), _COMPONENT_CONFIG_0)
        self._write(str(configdir1 / ".torchxconfig"), _COMPONENT_CONFIG_1)

        defaults = load_sections(
            prefix="component", dirs=[str(configdir0), str(configdir1)]
        )

        self.assertDictEqual(
            {
                "dist.ddp": {
                    "j": "1x2",
                    "image": "barbaz",
                    "env": "A=B,C=D",
                    "script_args": "a b c --d=e --f g",
                },
                "utils.echo": {
                    "msg": "abracadabra",
                },
                "utils.touch": {"file": "tmp.txt"},
            },
            defaults,
        )

    def test_get_configs(self) -> None:
        configdir0 = Path(self.test_dir) / "test_load_component_defaults" / "0"
        configdir1 = Path(self.test_dir) / "test_load_component_defaults" / "1"
        configdir0.mkdir(parents=True, exist_ok=True)
        configdir1.mkdir(parents=True, exist_ok=True)
        self._write(str(configdir0 / ".torchxconfig"), _COMPONENT_CONFIG_0)
        self._write(str(configdir1 / ".torchxconfig"), _COMPONENT_CONFIG_1)
        dirs = [str(configdir0), str(configdir1)]

        self.assertDictEqual(
            {},
            get_configs(
                prefix="component",
                name="non-existent",
                dirs=dirs,
            ),
        )
        self.assertDictEqual(
            {
                "j": "1x2",
                "image": "barbaz",
                "env": "A=B,C=D",
                "script_args": "a b c --d=e --f g",
            },
            get_configs(
                prefix="component",
                name="dist.ddp",
                dirs=dirs,
            ),
        )

    def test_get_config(self) -> None:
        configdir0 = Path(self.test_dir) / "test_load_component_defaults" / "0"
        configdir1 = Path(self.test_dir) / "test_load_component_defaults" / "1"
        configdir0.mkdir(parents=True, exist_ok=True)
        configdir1.mkdir(parents=True, exist_ok=True)
        self._write(str(configdir0 / ".torchxconfig"), _COMPONENT_CONFIG_0)
        self._write(str(configdir1 / ".torchxconfig"), _COMPONENT_CONFIG_1)
        dirs = [str(configdir0), str(configdir1)]

        self.assertEqual(
            "1x2",
            get_config(prefix="component", name="dist.ddp", key="j", dirs=dirs),
        )
        self.assertEqual(
            "barbaz",
            get_config(prefix="component", name="dist.ddp", key="image", dirs=dirs),
        )
        self.assertIsNone(
            get_config(prefix="component", name="dist.ddp", key="badkey", dirs=dirs),
        )
        self.assertIsNone(
            get_config(prefix="component", name="badname", key="j", dirs=dirs),
        )
        self.assertIsNone(
            get_config(prefix="badprefix", name="dist.ddp", key="j", dirs=dirs),
        )

    def test_load(self) -> None:
        cfg = {}
        load(scheduler="local_cwd", f=StringIO(_CONFIG), cfg=cfg)
        self.assertEqual("/home/bob/logs", cfg.get("log_dir"))
        self.assertEqual(True, cfg.get("prepend_cwd"))

    def test_no_override_load(self) -> None:
        cfg: Dict[str, CfgVal] = {"log_dir": "/foo/bar", "debug": 1}

        load(scheduler="local_cwd", f=StringIO(_CONFIG), cfg=cfg)
        self.assertEqual("/foo/bar", cfg.get("log_dir"))
        self.assertEqual(1, cfg.get("debug"))
        self.assertEqual(True, cfg.get("prepend_cwd"))

    @patch(
        TORCHX_GET_SCHEDULERS,
        return_value={"test": TestScheduler()},
    )
    def test_apply_default(self, _) -> None:
        with patch(PATH_CWD, return_value=Path(self.test_dir)):
            cfg: Dict[str, CfgVal] = {"s": "runtime_value"}
            apply(scheduler="test", cfg=cfg)

            self.assertEqual("runtime_value", cfg.get("s"))
            self.assertEqual(50, cfg.get("i"))
            self.assertEqual(1.2, cfg.get("f"))

    @patch(
        TORCHX_GET_SCHEDULERS,
        return_value={"test": TestScheduler()},
    )
    def test_apply_dirs(self, _) -> None:
        cfg: Dict[str, CfgVal] = {"s": "runtime_value"}
        apply(
            scheduler="test",
            cfg=cfg,
            dirs=[str(Path(self.test_dir) / "home" / "bob"), self.test_dir],
        )
        self.assertEqual("runtime_value", cfg.get("s"))
        self.assertEqual(100, cfg.get("i"))
        self.assertEqual(1.2, cfg.get("f"))

    @patch(
        TORCHX_GET_SCHEDULERS,
        return_value={"test": TestScheduler()},
    )
    def test_apply_from_annotated_config(self, _) -> None:
        def mock_getenv(x: str, y: Optional[str] = None) -> Optional[str]:
            if x != "TORCHX_CONFIG":
                return os.environ.get(x, y)
            else:
                return f"{self.test_dir}/another_torchx_config"

        with patch(PATH_CWD, return_value=Path(self.test_dir)):
            with patch(
                "torchx.runner.config.os.getenv",
                mock_getenv,
            ):
                cfg: Dict[str, CfgVal] = {"s": "runtime_value"}
                apply(
                    scheduler="test",
                    cfg=cfg,
                    # these dirs will be ignored
                    dirs=[str(Path(self.test_dir) / "home" / "bob"), self.test_dir],
                )
                self.assertEqual("runtime_value", cfg.get("s"))
                self.assertEqual(200, cfg.get("i"))
                self.assertEqual(None, cfg.get("f"))

    def test_dump_invalid_scheduler(self) -> None:
        with self.assertRaises(ValueError):
            dump(f=StringIO(), schedulers=["does-not-exist"])

    @patch(
        TORCHX_GET_SCHEDULERS,
        return_value={"test": TestScheduler()},
    )
    def test_dump_only_required(self, _) -> None:
        sfile = StringIO()

        # test scheduler has no required options hence expect empty string
        dump(f=sfile, required_only=True)

        cfg = {}
        sfile.seek(0)
        load(scheduler="test", f=sfile, cfg=cfg)

        # empty
        self.assertEqual({}, cfg)

    @patch(
        TORCHX_GET_SCHEDULERS,
        return_value={"test": TestScheduler()},
    )
    def test_load_invalid_runopt(self, _) -> None:
        cfg = {}
        load(
            scheduler="test",
            f=StringIO(_CONFIG_INVALID),
            cfg=cfg,
        )
        # options in the config file but not in runopts
        # should be ignored (we shouldn't throw an error since
        # this makes things super hard to guarantee BC - stale config file will fail
        # to run, we don't want that)

        self.assertEquals("option_that_exists", cfg.get("s"))

    def test_load_no_section(self) -> None:
        cfg = {}
        load(
            scheduler="local_cwd",
            f=StringIO(),
            cfg=cfg,
        )
        # is empty
        self.assertEqual({}, cfg)

        load(
            scheduler="local_cwd",
            f=StringIO("[scheduler_args.local_cwd]\n"),
            cfg=cfg,
        )
        # still empty
        self.assertEqual({}, cfg)

    @patch(
        TORCHX_GET_SCHEDULERS,
        return_value={"test": TestScheduler()},
    )
    def test_dump_and_load_all_runopt_types(self, _) -> None:
        sfile = StringIO()
        dump(sfile)

        sfile.seek(0)

        cfg = {}
        load(scheduler="test", f=sfile, cfg=cfg)

        # all runopts in the TestScheduler have defaults, just check against those
        for opt_name, opt in TestScheduler().run_opts():
            self.assertEqual(cfg.get(opt_name), opt.default)

    def test_dump_and_load_all_registered_schedulers(self) -> None:
        # dump all the runopts for all registered schedulers
        # load them back as run cfg
        # checks that all scheduler run options are accounted for

        sfile = StringIO()
        dump(sfile)

        for sched_name, sched in get_schedulers(session_name="_").items():
            sfile.seek(0)  # reset the file pos
            cfg = {}
            load(scheduler=sched_name, f=sfile, cfg=cfg)

            for opt_name, _ in sched.run_opts():
                self.assertTrue(opt_name in cfg)
