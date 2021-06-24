# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import glob
import importlib
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from inspect import getmembers, isfunction
from types import ModuleType
from typing import Dict, List, Optional, Union, Callable

from pyre_extensions import none_throws
from torchx.specs import AppDef
from torchx.specs.file_linter import get_fn_docstring, validate
from torchx.util import entrypoints
from torchx.util.io import read_conf_file

logger: logging.Logger = logging.getLogger(__name__)


@dataclass
class _Component:
    """
    Definition of the component

    Args:
        name: The name of the component, which usually MODULE_PATH.FN_NAME
        module_name: Full module name, equivalent to ``module.__name__``
        description: The description of the component, taken from the desrciption
            of the function that creates component
        group: Logic group of the component
        fn_name: Function name that creates component
        fn: Function that creates component
    """

    name: str
    module_name: str
    description: str
    group: str
    fn_name: str
    fn: Callable[..., AppDef]


class ComponentsFinder(abc.ABC):
    @abc.abstractmethod
    def find(self) -> List[_Component]:
        """
        Retrieves a set of components. A component is defined as a python
        function that conforms to ``torchx.specs.file_linter`` linter.

        Returns:
            List of components
        """

    def _validate_and_get_description(
        self, module: ModuleType, function_name: str
    ) -> Optional[str]:
        module_path = os.path.abspath(module.__file__)
        source = read_conf_file(module_path)
        if len(validate(source, torchx_function=function_name)) != 0:
            return None
        func_definition, _ = none_throws(get_fn_docstring(source, function_name))
        return func_definition


class ModuleComponentsFinder(ComponentsFinder):
    """Retrieves components from the directory associated with module.

    Finds all components inside a directory associated with the module path in a recursive
    manner. Class finds the lowest level directory for the ``module.__file`` and uses it
    as a base dir. Then it recursively traverses the base dir to find all modules. For each
    module it tries to find a component, which is a well-defined python function that
    returns ``torchx.specs.AppDef``.

    ``group`` can be supplied, which is used to construct component name and specify
    the logical grouping of the components.

    ::
     module = "main.foo.bar" # resolves to main/foo/bar.py
     # main.foo will be replaced by the "trainer" in component.name
     components = ModuleComponentsFinder(module, alias = "trainer").find()


    """

    def __init__(self, module: Union[str, ModuleType], group: str) -> None:
        self._module = module
        self._group = group

    def find(self) -> List[_Component]:
        module = self._try_load_module(self._module)
        dir_name = os.path.dirname(module.__file__)
        return self._get_components_from_dir(
            dir_name, self._get_base_module_name(module)
        )

    def _get_base_module_name(self, module: ModuleType) -> str:
        filepath = module.__file__
        if filepath.endswith("__init__.py"):
            return module.__name__
        else:
            return module.__name__.rsplit(".", 1)[0]

    def _try_load_module(self, module: Union[str, ModuleType]) -> ModuleType:
        if isinstance(module, str):
            return importlib.import_module(module)
        else:
            return module

    def _get_components_from_dir(
        self, search_dir: str, base_module: str
    ) -> List[_Component]:
        search_pattern = os.path.join(search_dir, "**", "*.py")
        component_defs = []
        for filepath in glob.glob(search_pattern, recursive=True):
            module = self._try_load_module(
                self._get_module_name(filepath, search_dir, base_module)
            )
            defs = self._get_components_from_module(base_module, module)
            component_defs += defs
        return component_defs

    def _is_private_function(self, function_name: str) -> bool:
        return function_name.startswith("_")

    def _get_component_name(
        self, base_module: str, module_name: str, fn_name: str
    ) -> str:
        if self._group is not None:
            module_name = module_name.replace(base_module, none_throws(self._group), 1)
            if module_name.startswith("."):
                module_name = module_name[1:]
        return f"{module_name}.{fn_name}"

    def _get_components_from_module(
        self, base_module: str, module: ModuleType
    ) -> List[_Component]:
        functions = getmembers(module, isfunction)
        component_defs = []
        for function_name, function in functions:
            component_desc = self._validate_and_get_description(module, function_name)
            if self._is_private_function(function_name) or not component_desc:
                continue
            component_def = _Component(
                name=self._get_component_name(
                    base_module, module.__name__, function_name
                ),
                module_name=module.__name__,
                description=component_desc,
                group=self._group,
                fn_name=function_name,
                fn=function,
            )
            component_defs.append(component_def)
        return component_defs

    def _strip_init(self, module_name: str) -> str:
        if module_name.endswith(".__init__"):
            return module_name.rsplit(".__init__")[0]
        elif module_name == "__init__":
            return ""
        else:
            return module_name

    def _get_module_name(self, filepath: str, search_dir: str, base_module: str) -> str:
        module_path = os.path.relpath(filepath, search_dir)
        module_path, _ = os.path.splitext(module_path)
        module_name = module_path.replace(os.path.sep, ".")
        module_name = self._strip_init(module_name)
        if not module_name:
            return base_module
        else:
            return f"{base_module}.{module_name}"


def _load_components() -> Dict[str, _Component]:
    component_modules = entrypoints.load_group("torchx.components", default={})
    component_defs: OrderedDict[str, _Component] = OrderedDict()

    finder = ModuleComponentsFinder("torchx.components", "")
    for component in finder.find():
        component_defs[component.name] = component

    for component_group, component_module in component_modules.items():
        finder = ModuleComponentsFinder(component_module, component_group)
        found_components = finder.find()
        for component in found_components:
            component_defs[component.name] = component
    return component_defs


_components: Optional[Dict[str, _Component]] = None


def get_components() -> Dict[str, _Component]:
    """
    Returns all components registered in the [torchx.components] entry_points.txt
    Each line is a key, value pair in a format:
    ::
      foo = test.bar

    Where ``test.bar`` is a valid path to the python module and ``foo`` is alias.
    Note: the module mast be discoverable by the torchx. A

    Returns:
        Components in a format : {ALIAS: LIST_OF_COMPONENTS}
    """
    global _components
    if not _components:
        _components = _load_components()
    return none_throws(_components)


def get_component(name: str) -> Optional[_Component]:
    """
    Retrieves components by the provided name.

    Returns:
        Component or None if no component with ``name`` exists
    """
    components = get_components()
    return components.get(name, None)
