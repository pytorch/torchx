#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import tempfile
import unittest
from contextlib import contextmanager
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Mapping, Optional
from unittest.mock import patch

from torchx.config import (
    ConfigData,
    get_config_store,
    IniConfigStore,
    seed_config_store_with_schedulers,
)
from torchx.config.config import _find_config_files, _query_file
from torchx.schedulers import get_schedulers, Scheduler
from torchx.schedulers.api import DescribeAppResponse, Stream
from torchx.specs import AppDef, AppDryRunInfo, CfgVal, runopts


class ConfigDataTest(unittest.TestCase):
    def test_apply_to(self) -> None:
        props: Dict[str, CfgVal] = {"prop1": "val1"}
        config = ConfigData({"prop1": "new_val1", "prop2": "val2"}, "test_component")

        new_config = config.apply_to(props)
        self.assertEqual(new_config, {"prop1": "val1", "prop2": "val2"})


CONFIG_COMPONENTS_1 = """#
[components:config1]
foo = baz
hello = world

[other:config1]
foo = bar
param1 = value1


"""

CONFIG_COMPONENTS_2 = """#
[components:config2]
foo = bar
hello = world

[other:config2]
param2 = value2

"""

PATH_CWD = "torchx.config.config.Path.cwd"
FIND_CONFIG = "torchx.config.config._find_config_files"
TORCHX_GET_SCHEDULERS = "torchx.config.config.get_schedulers"
GET_ENV = "torchx.config.config.getenv"


@contextmanager
def file_content(
    content: str, suffix: Optional[str] = None
) -> Generator[Path, None, None]:
    test_dir = tempfile.mkdtemp(prefix="torchx_config_test")
    if suffix:
        filename = f".torchxconfig.{suffix}"
    else:
        filename = ".torchxconfig"
    f = Path(test_dir) / filename
    with open(f, "w") as fp:
        fp.write(content)
        fp.flush()
        yield f


class IniConfigStoreTest(unittest.TestCase):
    def test_get_all(self) -> None:
        with file_content(CONFIG_COMPONENTS_1) as config_file:
            with patch(FIND_CONFIG, return_value=[str(config_file)]):
                config_store = IniConfigStore([str(config_file.parent)])
                configs = config_store.get_all("components")
                self.assertEqual(1, len(configs))
                self.assertEqual("config1", configs[0].coll_name)
                self.assertEqual({"foo": "baz", "hello": "world"}, configs[0])

    def test_get(self) -> None:
        with file_content(CONFIG_COMPONENTS_1) as config_file:
            with patch(FIND_CONFIG, return_value=[str(config_file)]):
                config_store = IniConfigStore([str(config_file.parent)])
                config = config_store.get("config1", "components")
                self.assertEqual("config1", config.coll_name)
                self.assertEqual({"foo": "baz", "hello": "world"}, config)

    def test_get_key(self) -> None:
        with file_content(CONFIG_COMPONENTS_1) as config_file:
            with patch(FIND_CONFIG, return_value=[str(config_file)]):
                config_store = IniConfigStore([str(config_file.parent)])
                val = config_store.get_key("config1", "hello", "components")
                self.assertEqual(val, "world")

    def test_get_key_from_multiple_config_files(self) -> None:
        with file_content(CONFIG_COMPONENTS_1) as config_file_1:
            with file_content(CONFIG_COMPONENTS_2) as config_file_2:
                with patch(
                    FIND_CONFIG,
                    return_value=[str(config_file_1), str(config_file_2)],
                ):
                    config_store = IniConfigStore(
                        [str(config_file_1.parent), str(config_file_2.parent)]
                    )
                    val1 = config_store.get_key("config1", "foo", "components")
                    self.assertEqual(val1, "baz")
                    val2 = config_store.get_key("config2", "foo", "components")
                    self.assertEqual(val2, "bar")

    def test_get_key_from_default_location(self) -> None:
        with file_content(CONFIG_COMPONENTS_1) as config_file:
            with patch(PATH_CWD, return_value=config_file.parent):
                config_store = IniConfigStore([])
                val = config_store.get_key("config1", "hello", "components")
                self.assertEqual(val, "world")

    def test_get_key_from_default_location_with_suffix(self) -> None:
        with file_content(CONFIG_COMPONENTS_1, suffix="external") as config_file:
            with patch(PATH_CWD, return_value=config_file.parent):
                config_store = IniConfigStore([], suffix="external")
                val = config_store.get_key("config1", "hello", "components")
                self.assertEqual(val, "world")

    def test_find_config_file(self) -> None:
        with file_content(CONFIG_COMPONENTS_1) as config_file:
            file_path = _find_config_files([str(Path(config_file).parent)])
            self.assertEqual(file_path, [config_file])

    def test_find_config_file_with_suffix(self) -> None:
        with file_content(CONFIG_COMPONENTS_1, suffix="external") as config_file:
            file_path = _find_config_files(
                [str(Path(config_file).parent)], suffix="external"
            )
            self.assertEqual(file_path, [config_file])

    def test_init_store(self) -> None:
        test_dir = tempfile.mkdtemp(prefix="torchx_config_test")
        test_config_file = f"{test_dir}/.torchxconfig"

        config_1 = ConfigData(
            {"k1": "v1", "k2": "v2"}, "test_component_1", label="component"
        )

        config_2 = ConfigData(
            {"k3": "v3", "k4": "v4"}, "test_component_2", label="component"
        )

        with open(test_config_file, "w") as f:
            IniConfigStore.init_store(
                f,
                [
                    config_1,
                    config_2,
                ],
            )
        store = IniConfigStore([(str(test_dir))])
        stored_configs = store.get_all("component")
        self.assertEqual(len(stored_configs), 2)
        self.assertEqual(stored_configs[0], config_1)
        self.assertEqual(stored_configs[1], config_2)

        shutil.rmtree(test_dir)


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


class ConfigModuleTest(unittest.TestCase):
    def test_dump_and_load_all_registered_schedulers(self) -> None:
        # dump all the runopts for all registered schedulers
        # load them back as run cfg
        # checks that all scheduler run options are accounted for

        sfile = StringIO()
        seed_config_store_with_schedulers(sfile)

        for sched_name, sched in get_schedulers(session_name="_").items():
            sfile.seek(0)  # reset the file pos

            config_data = _query_file(
                sfile, collection_filter=lambda c: c == sched_name
            )
            self.assertEqual(len(config_data), 1)
            config_data = config_data[0]
            for opt_name, _ in sched.run_opts():
                self.assertTrue(opt_name in config_data)

    def test_dump_invalid_scheduler(self) -> None:
        with self.assertRaises(ValueError):
            seed_config_store_with_schedulers(
                f=StringIO(), schedulers=["does-not-exist"]
            )

    @patch(
        TORCHX_GET_SCHEDULERS,
        return_value={"test": TestScheduler()},
    )
    def test_dump_only_required(self, _) -> None:
        sfile = StringIO()

        # test scheduler has no required options hence expect empty string
        seed_config_store_with_schedulers(f=sfile, required_only=True)

        sfile.seek(0)
        config_data = _query_file(sfile, collection_filter=lambda c: c == "test")
        self.assertEqual(1, len(config_data))
        self.assertEqual(0, len(config_data[0]))

    @patch(
        TORCHX_GET_SCHEDULERS,
        return_value={"test": TestScheduler()},
    )
    def test_dump_and_load_all_runopt_types(self, _) -> None:
        sfile = StringIO()
        seed_config_store_with_schedulers(f=sfile)

        sfile.seek(0)

        type_hints = {
            param_key: opts.opt_type for param_key, opts in TestScheduler().run_opts()
        }
        config_data = _query_file(
            sfile, collection_filter=lambda c: c == "test", type_hints=type_hints
        )

        # all runopts in the TestScheduler have defaults, just check against those
        for opt_name, opt in TestScheduler().run_opts():
            self.assertEqual(config_data[0][opt_name], opt.default)

    def test_get_config_store_env_var(self) -> None:
        with file_content(CONFIG_COMPONENTS_1) as config_file:
            with patch(GET_ENV, return_value=str(config_file)):
                config_store = get_config_store(dirs=["/tmp/foo"])
                val = config_store.get_key("config1", key="hello", label="components")
                self.assertEqual(val, "world")
