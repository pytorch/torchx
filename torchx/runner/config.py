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


def dump(
    f: TextIO, schedulers: Optional[List[str]] = None, required_only: bool = False
) -> None:
    """
    Dumps a default INI-style config template containing the runopts for the
    given scheduler names into ``f``. If no ``schedulers`` are specified
    dumps all known registered schedulers.

    Optional runopts are pre-filled  with their default values.
    Required runopts are set with a ``<FIXME_...>`` placeholder.
    Each scheduler's runopts are written in the section called
    ``[default.scheduler_args.{scheduler_name}]`` (e.g. ``[default.scheduler_args.kubernetes]``)

    To only dump required runopts pass ``required_only=True``.

    Raises a ``ValueError`` if given a scheduler name that is not known
    """

    if schedulers:
        scheds = schedulers
    else:
        scheds = get_schedulers(session_name="_").keys()

    config = _configparser()
    for sched_name in scheds:
        sched = _get_scheduler(sched_name)

        section = f"default.scheduler_args.{sched_name}"
        config.add_section(section)

        for opt_name, opt in sched.run_opts():
            if opt.is_required:
                val = f"<FIXME_WITH_A_{get_type_name(opt.opt_type)}_VALUE>"
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


def apply(profile: str, scheduler: str, runcfg: RunConfig) -> None:
    """
    Loads .torchxconfig files from predefined locations according
    to a load hierarchy and applies the loaded configs into the
    given ``runcfg``. The load hierarchy is as follows (in order of precedence):

    #. ``runcfg`` given to this function
    #. configs loaded from ``$HOME/.torchxconfig``
    #. configs loaded from ``$CWD/.torchxconfig``

    Note that load hierarchy does NOT overwrite, but rather adds.
    That is, the configs already present in ``runcfg`` are not
    overridden during the load.
    """
    lookup_dirs = [Path.home(), Path.cwd()]

    for d in lookup_dirs:
        configfile = d / ".torchxconfig"
        if configfile.exists():
            log.info(f"loading configs from {configfile}")
            with open(str(configfile), "r") as f:
                load(profile, scheduler, f, runcfg)


def load(profile: str, scheduler: str, f: TextIO, runcfg: RunConfig) -> None:
    """
    loads the section ``[{profile}.scheduler_args.{scheduler}]`` from the given
    configfile ``f`` (in .INI format) into the provided ``runcfg``, only adding
    configs that are NOT currently in the given ``runcfg`` (e.g. does not
    override existing values in ``runcfg``). If no section is found, does nothing.
    """

    config = _configparser()
    config.read_file(f)

    runopts = _get_scheduler(scheduler).run_opts()

    section = f"{profile}.scheduler_args.{scheduler}"
    if config.has_section(section):
        for name, value in config.items(section):
            if name in runcfg.cfgs:
                # DO NOT OVERRIDE existing configs
                continue

            if value == _NONE:
                # should map to None (not str 'None')
                # this also handles empty or None lists
                runcfg.set(name, None)
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
                        runcfg.set(name, config.getboolean(section, name))
                    elif runopt.opt_type is List[str]:
                        runcfg.set(name, value.split(";"))
                    else:
                        # pyre-ignore[29]
                        runcfg.set(name, runopt.opt_type(value))
