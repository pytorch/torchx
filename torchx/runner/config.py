#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import configparser as configparser
import logging
from pathlib import Path
from typing import List, Optional, TextIO

from torchx.schedulers import Scheduler, get_schedulers
from torchx.specs import RunConfig, get_type_name
from torchx.specs.api import runopt


_NONE = "None"

log: logging.Logger = logging.getLogger(__name__)


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


def dump(
    f: TextIO, schedulers: Optional[List[str]] = None, required_only: bool = False
) -> None:
    """
    Dumps a default INI-style config template containing the :py:class:torchx.specs.runopts for the
    given scheduler names into the file-like object specified by ``f``.
    If no ``schedulers`` are specified dumps all known registered schedulers.

    Optional runopts are pre-filled  with their default values.
    Required runopts are set with a ``FIXME: ...`` placeholder.
    To only dump required runopts pass ``required_only=True``.

    Each scheduler's runopts are written in the section called
    ``[default.{scheduler_name}.cfg]``.

    For example:

    ::

     [kubernetes]
     namespace = default
     queue = #FIXME (str)Volcano queue to schedule job in

    Raises:
        ``ValueError`` - if given a scheduler name that is not known
    """

    if schedulers:
        scheds = schedulers
    else:
        scheds = get_schedulers(session_name="_").keys()

    config = _configparser()
    for sched_name in scheds:
        sched = _get_scheduler(sched_name)

        section = f"{sched_name}"
        config.add_section(section)

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

            config.set(section, opt_name, val)
    config.write(f, space_around_delimiters=True)


def apply(scheduler: str, cfg: RunConfig, dirs: Optional[List[str]] = None) -> None:
    """
    Loads a ``.torchxconfig`` INI file from the specified directories in
    preceding order and applies the run configs for the scheduler onto
    the given ``cfg``.

    If no ``dirs`` is specified, then it looks for ``.torchxconfig`` in the
    current working directory. If a specified directory does not have ``.torchxconfig``
    then it is ignored.

    Note that the configs already present in the given ``cfg`` take precedence
    over the ones in the config file and only new configs are added. The same holds
    true for the configs loaded in list order.

    For instance if ``cfg = {"foo": "bar"}`` and the config file is:

    ::

     # dir_1/.torchxconfig
     [local_cwd]
     foo = baz
     hello = world

    # dir_2/.torchxconfig
    [local_cwd]
    hello = bob


    Then after the method call, ``cfg = {"foo": "bar", "hello": "world"}``.
    """

    if not dirs:
        dirs = [str(Path.cwd())]

    for d in dirs:
        configfile = Path(d) / ".torchxconfig"
        if configfile.exists():
            log.info(f"loading configs from {configfile}")
            with open(str(configfile), "r") as f:
                load(scheduler, f, cfg)


def load(scheduler: str, f: TextIO, cfg: RunConfig) -> None:
    """
    loads the section ``[{scheduler}]`` from the given
    configfile ``f`` (in .INI format) into the provided ``runcfg``, only adding
    configs that are NOT currently in the given ``runcfg`` (e.g. does not
    override existing values in ``runcfg``). If no section is found, does nothing.
    """

    config = _configparser()
    config.read_file(f)

    runopts = _get_scheduler(scheduler).run_opts()

    section = f"{scheduler}"
    if config.has_section(section):
        for name, value in config.items(section):
            if name in cfg.cfgs:
                # DO NOT OVERRIDE existing configs
                continue

            if value == _NONE:
                # should map to None (not str 'None')
                # this also handles empty or None lists
                cfg.set(name, None)
            else:
                runopt = runopts.get(name)

                if runopt is None:
                    log.warning(
                        f"`{name} = {value}` was declared in the [{section}] section "
                        f" of the config file but is not a runopt of `{scheduler}` scheduler."
                        f" Remove the entry from the config file to no longer see this warning"
                    )
                else:
                    if runopt.opt_type is bool:
                        # need to handle bool specially since str -> bool is based on
                        # str emptiness not value (e.g. bool("False") == True)
                        cfg.set(name, config.getboolean(section, name))
                    elif runopt.opt_type is List[str]:
                        cfg.set(name, value.split(";"))
                    else:
                        # pyre-ignore[29]
                        cfg.set(name, runopt.opt_type(value))
