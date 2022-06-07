# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Status: Beta

You can store the scheduler run cfg (run configs) for your project
by storing them in the ``.torchxconfig`` file. Currently this file is only read
and honored when running the component from the CLI.

CLI Usage
~~~~~~~~~~~


**Scheduler Config**

#. ``cd`` into the directory where you want the ``.torchxconfig`` file to be dropped.
   The CLI only picks up ``.torchxconfig`` files from the current-working-directory (CWD)
   so chose a directory where you typically run ``torchx`` from. Typically this
   is the root of your project directory.

#. Generate the config file by running

   .. code-block:: shell-session

    $ torchx configure -s <comma,delimited,scheduler,names>

    # -- or for all registered schedulers --
    $ torchx configure

#. If you specified ``-s local_cwd,kubernetes``, you should see a ``.torchxconfig``
   file as shown below:

   .. code-block:: shell-session

    $ cat .torchxconfig
    [local_cwd]

    [kubernetes]
    queue = #FIXME:(str) Volcano queue to schedule job in

#. ``.torchxconfig`` in in INI format and the section names map to the scheduler names.
   Each section contains the run configs for the scheduler as ``$key = $value`` pairs.
   You may find that certain schedulers have empty sections, this means that
   the scheduler defines sensible defaults for all its run configs hence no run configs
   are required at runtime. If you'd like to override the default you can add them.
   **TIP:** To see all the run options for a scheduler use ``torchx runopts <scheduler_name>``.

#. The sections with ``FIXME`` placeholders are run configs that are required
   by the scheduler. Replace these with the values that apply to you.

#. **IMPORTANT:** If you are happy with the scheduler provided defaults for a particular
   run config, you **should not** redundantly specity them in ``.torchxconfig`` with the
   same default value. This is because the scheduler may decide to change the default
   value at a later date which would leave you with a stale default.

#. Now you can run your component without having to specify the scheduler run configs
   each time. Just make sure the directory you are running ``torchx`` cli from actually
   has ``.torchxconfig``!

   .. code-block:: shell-session

    $ ls .torchxconfig
    .torchxconfig

    $ torchx run -s local_cwd ./my_component.py:train

#. In addition, it is possible to specify a different config other than .torchxconfig to
   load at runtime. Requirements are that the config path is specified by enviornment
   variable TORCHX_CONFIG. It also disables hierarchy loading configs from multiple
   directories as the cases otherwise.

**Component Config**

You can specify component defaults by adding a section prefixed with
``component:``.

.. code-block:: ini

    [component:dist.ddp]
    j=2x8
    cpu=4

Now when you run the ``dist.ddp`` component those configs are automatically
picked up.

.. code-block:: shell-session

    $ torchx run -s local_cwd dist.ddp
    ... runs with -j 2x8 --cpu 4


**CLI Subcommand Config**

The default arguments for the ``torchx`` subcommands can be overwritten. Any
``--foo FOO`` argument can be set via the correspond ``[cli:<cmd>]`` settings
block.

For the ``run`` command you can additionally set ``component`` to set the
default component to run.

.. code-block:: ini

    [cli:run]
    component=dist.ddp
    scheduler=local_docker
    workspace=file://some_workspace


Programmatic Usage
~~~~~~~~~~~~~~~~~~~

Unlike the cli, ``.torchxconfig`` file **is not** picked up automatically
from ``CWD`` if you are programmatically running your component with :py:class:`torchx.runner.Runner`.
You'll have to manually specify the directory containing ``.torchxconfig``.

Below is an example

.. doctest:: [runner_config_example]

 from torchx.runner import get_runner
 from torchx.runner.config import apply
 import torchx.specs as specs

 def my_component(a: int) -> specs.AppDef:
    # <... component body omitted for brevity ...>
    pass

 scheduler = "local_cwd"
 cfg = {"log_dir": "/these/take/outmost/precedence"}

 apply(scheduler, cfg, dirs=["/home/bob"])  # looks for /home/bob/.torchxconfig
 get_runner().run(my_component(1), scheduler, cfg)

You may also specify multiple directories (in preceding order) which is useful when
you want to keep personal config overrides on top of a project defined default.

"""
import collections
import configparser
import logging
from abc import ABC, abstractmethod
from os import getenv
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, TextIO, Type

from torchx.schedulers import get_schedulers, Scheduler
from torchx.specs import CfgVal, get_type_name
from torchx.specs.api import runopt

log: logging.Logger = logging.getLogger(__name__)


ALWAYS_PASS_FN: Callable[[str], bool] = lambda x: True
CONFIG_FILE = ".torchxconfig"
CONFIG_PREFIX_DELIM = ":"
ENV_TORCHX_CONFIG = "TORCHXCONFIG"
_NONE = "None"


class ConfigData(collections.abc.Mapping):
    """
    Dict-like configuration data.
    """

    def __init__(
        self, key_vals: Dict[str, CfgVal], coll_name: str, label: Optional[str] = None
    ) -> None:
        self.coll_name = coll_name
        self.label = label
        self._dict = key_vals

    def __getitem__(self, item: str) -> CfgVal:
        return self._dict.__getitem__(item)

    def __iter__(self) -> Iterator[CfgVal]:
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def apply_to(self, cfg: Dict[str, CfgVal]) -> Dict[str, CfgVal]:
        dict_copy = self._dict.copy()
        dict_copy.update(cfg)
        return dict_copy


class ConfigStore(ABC):
    @abstractmethod
    def get_all(
        self,
        label: Optional[str] = None,
        type_hints: Optional[Dict[str, Type[CfgVal]]] = None,
    ) -> List[ConfigData]:
        ...

    @abstractmethod
    def get(
        self,
        collection: str,
        label: Optional[str] = None,
        type_hints: Optional[Dict[str, Type[CfgVal]]] = None,
    ) -> ConfigData:
        ...

    @abstractmethod
    def get_key(
        self,
        collection: str,
        key: str,
        label: Optional[str] = None,
        type_hints: Optional[Dict[str, Type[CfgVal]]] = None,
    ) -> Optional[CfgVal]:
        ...


def _find_config_files(dirs: List[str], suffix: str = "") -> List[str]:
    config_files = []
    for d in dirs:
        file_name = CONFIG_FILE + "." + suffix if suffix else CONFIG_FILE
        configfile = Path(d) / file_name
        if configfile.exists():
            config_files.append(configfile)
    return config_files


def _configparser() -> configparser.ConfigParser:
    """
    Sets up the configparser and returns it. The same config parser
    should be used between dumps() and loads() methods for ser/de compatibility
    """

    config = configparser.ConfigParser()
    # if optionxform is not overridden, configparser will by default lowercase
    # the option keys because it is compatible with Windows INI files
    # which are expected to be parsed case insensitive.
    # override since torchx's runopts are case-sensitive
    # see: https://stackoverflow.com/questions/19359556/configparser-reads-capital-keys-and-make-them-lower-case
    # pyre-ignore[8]
    config.optionxform = lambda option: option

    return config


def _strip_prefix(prefix: str, section_name: str) -> Optional[str]:
    # returns the section_name with the prefix removed
    # or None if the section name does not match the prefix
    idx = section_name.find(CONFIG_PREFIX_DELIM)

    # index guard (e.g ":foo" and "foo:" are bad, just return None)
    if 0 < idx < len(section_name) - 1:
        if section_name[0:idx] == prefix:
            return section_name[idx + 1 :]
    return None


def _query_file(
    f: TextIO,
    collection_filter: Callable[[str], bool] = ALWAYS_PASS_FN,
    key_filter: Callable[[str], bool] = ALWAYS_PASS_FN,
    label: Optional[str] = None,
    type_hints: Optional[Dict[str, Type[CfgVal]]] = None,
) -> List[ConfigData]:
    section_configs = []

    config = _configparser()
    config.read_file(f)
    for section_name in config.sections():
        if label:
            coll_name = _strip_prefix(label, section_name)
        else:
            coll_name = section_name
        if coll_name and collection_filter(coll_name):
            section_config = {}
            for key, value in config.items(section_name):
                if key not in section_config and key_filter(key):
                    if value == _NONE:
                        # should map to None (not str 'None')
                        # this also handles empty or None lists
                        section_config[key] = None
                    elif type_hints:
                        if key not in type_hints:
                            continue
                        _type: Type[CfgVal] = type_hints[key]
                        if _type is bool:
                            section_config[key] = config.getboolean(section_name, key)
                        elif _type is List[str]:
                            section_config[key] = value.split(";")
                        else:
                            # pyre-ignore
                            section_config[key] = _type(value)
                    else:
                        section_config[key] = value
            config_data = ConfigData(section_config, coll_name=coll_name, label=label)
            section_configs.append(config_data)
    return section_configs


class IniConfigStore(ConfigStore):
    def __init__(
        self,
        dirs: Optional[List[str]] = None,
        config_file: Optional[str] = None,
        suffix: str = "",
    ) -> None:
        self.config_file: Optional[str] = config_file
        if dirs:
            self.dirs: List[str] = dirs
        else:
            cwd = str(Path.cwd())
            self.dirs: List[str] = [cwd]
            log.info(f"Using cwd '{cwd}'' for loading config file ")
        self.suffix = suffix

    @classmethod
    def init_store(cls, f: TextIO, config_data: List[ConfigData]) -> None:
        config = _configparser()
        for c in config_data:
            section_name = (
                f"{c.label}{CONFIG_PREFIX_DELIM}{c.coll_name}"
                if c.label
                else c.coll_name
            )
            config.add_section(section_name)
            for k, v in c.items():
                config.set(section_name, k, v)
        config.write(f, space_around_delimiters=True)

    def _query(
        self,
        collection_filter: Callable[[str], bool] = ALWAYS_PASS_FN,
        key_filter: Callable[[str], bool] = ALWAYS_PASS_FN,
        label: Optional[str] = None,
        type_hints: Optional[Dict[str, Type[CfgVal]]] = None,
    ) -> List[ConfigData]:
        configs = []
        if self.config_file:
            assert Path(
                self.config_file
            ).exists(), f"{str(self.config_file)} expected but not found"
            config_files = [str(self.config_file)]

        else:
            config_files = _find_config_files(self.dirs, self.suffix)
        for f in config_files:
            with open(f, "r") as fd:
                config_data = _query_file(
                    fd, collection_filter, key_filter, label, type_hints
                )
                if config_data:
                    configs.extend(config_data)
        return configs

    def get_all(
        self,
        label: Optional[str] = None,
        type_hints: Optional[Dict[str, Type[CfgVal]]] = None,
    ) -> List[ConfigData]:
        return self._query(
            collection_filter=ALWAYS_PASS_FN,
            key_filter=ALWAYS_PASS_FN,
            label=label,
            type_hints=type_hints,
        )

    def get(
        self,
        collection: str,
        label: Optional[str] = None,
        type_hints: Optional[Dict[str, Type[CfgVal]]] = None,
    ) -> ConfigData:
        colls = self._query(
            collection_filter=lambda c: c == collection,
            key_filter=ALWAYS_PASS_FN,
            label=label,
            type_hints=type_hints,
        )
        if colls:
            return colls[0]
        return ConfigData({}, collection, label=label)

    def get_key(
        self,
        collection: str,
        key: str,
        label: Optional[str] = None,
        type_hints: Optional[Dict[str, Type[CfgVal]]] = None,
    ) -> Optional[CfgVal]:
        colls = self._query(
            collection_filter=lambda c: c == collection,
            key_filter=lambda k: k == key,
            label=label,
            type_hints=type_hints,
        )
        if colls and key in colls[0]:
            return colls[0][key]


def _get_scheduler(name: str) -> Scheduler:
    schedulers = get_schedulers(session_name="_")
    sched = schedulers.get(name)
    if not sched:
        raise ValueError(
            f"`{name}` is not a registered scheduler. Valid scheduler names: {schedulers.keys()}"
        )
    return sched


def _fixme_placeholder(runopt: runopt, max_len: int = 60) -> str:
    ph = f"#FIXME:({get_type_name(runopt.opt_type)}) {runopt.help}"
    return ph if len(ph) <= max_len else f"{ph[:max_len]}..."


def seed_config_store_with_schedulers(
    f: TextIO, schedulers: Optional[List[str]] = None, required_only: bool = False
) -> None:
    configs = []
    if schedulers:
        scheds = schedulers
    else:
        scheds = get_schedulers(session_name="_").keys()
    for sched_name in scheds:
        sched = _get_scheduler(sched_name)

        section = f"{sched_name}"
        kvs = {}

        for opt_name, opt in sched.run_opts():
            if opt.is_required:
                val = _fixme_placeholder(opt)
            else:  # not required runopts MUST have a default
                if required_only:
                    continue

                # serialize list elements with `;` delimiter (consistent with torchx cli)
                if opt.opt_type == List[str]:
                    # deal with empty or None default lists
                    if opt.default:
                        # pyre-ignore[6] opt.default type checked already as List[str]
                        val = ";".join(opt.default)
                    else:
                        val = _NONE
                else:
                    val = f"{opt.default}"

            kvs[opt_name] = val
        configs.append(ConfigData(kvs, section))
    IniConfigStore.init_store(f, configs)


def get_config_store(dirs: Optional[List[str]]) -> ConfigStore:
    config = getenv(ENV_TORCHX_CONFIG, "")
    if len(config) > 0:
        dirs = [config]
        log.info(f"Using {ENV_TORCHX_CONFIG} env variable for config file")
        config_store = IniConfigStore(config_file=config)
    else:
        config_store = IniConfigStore(dirs)
    return config_store
