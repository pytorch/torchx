# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import abc
import glob
import importlib
import inspect
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from inspect import getmembers, isfunction
from types import ModuleType
from typing import Callable, Dict, List, Optional, Union

from torchx.specs import AppDef
from torchx.specs.file_linter import get_fn_docstring, validate
from torchx.util import entrypoints
from torchx.util.io import read_conf_file

from torchx.util.types import none_throws


logger: logging.Logger = logging.getLogger(__name__)


class ComponentValidationException(Exception):
    pass


class ComponentNotFoundException(Exception):
    pass


@dataclass
class _Component:
    """
    Definition of the component

    Args:
        name: The name of the component, which usually MODULE_PATH.FN_NAME
        description: The description of the component, taken from the desrciption
            of the function that creates component. In case of no docstring, description
            will be the same as name
        fn_name: Function name that creates component
        fn: Function that creates component
        validation_errors: Validation errors
    """

    name: str
    description: str
    fn_name: str
    fn: Callable[..., AppDef]
    validation_errors: List[str]


class ComponentsFinder(abc.ABC):
    @abc.abstractmethod
    def find(self) -> List[_Component]:
        """
        Retrieves a set of components. A component is defined as a python
        function that conforms to ``torchx.specs.file_linter`` linter.

        Returns:
            List of components
        """

    def _get_module_source(self, module: ModuleType) -> str:
        module_path = os.path.abspath(module.__file__)
        return read_conf_file(module_path)


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
     components = ModuleComponentsFinder(module, group = "trainer").find()


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
        # pyre-fixme[16]: `Optional` has no attribute `endswith`.
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
            module_name = self._get_module_name(filepath, search_dir, base_module)
            # TODO(aivanou): move `torchx.components.base` to `torchx.specs`, since
            # there is nothing related to components in `torchx.components.base`
            # see https://github.com/pytorch/torchx/issues/261
            if module_name.startswith("torchx.components.base"):
                continue
            module = self._try_load_module(module_name)
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
        module_path = os.path.abspath(module.__file__)
        for function_name, function in functions:
            linter_errors = validate(module_path, function_name)
            component_desc, _ = get_fn_docstring(function)
            component_def = _Component(
                name=self._get_component_name(
                    base_module, module.__name__, function_name
                ),
                description=component_desc,
                fn_name=function_name,
                fn=function,
                validation_errors=[
                    linter_error.description for linter_error in linter_errors
                ],
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


class CustomComponentsFinder(ComponentsFinder):
    def __init__(self, filepath: str, function_name: str) -> None:
        self._filepath = filepath
        self._function_name = function_name

    def _get_validation_errors(self, path: str, function_name: str) -> List[str]:
        linter_errors = validate(path, function_name)
        return [linter_error.description for linter_error in linter_errors]

    def find(self) -> List[_Component]:
        validation_errors = self._get_validation_errors(
            self._filepath, self._function_name
        )

        file_source = read_conf_file(self._filepath)
        namespace = globals()
        exec(file_source, namespace)  # noqa: P204
        if self._function_name not in namespace:
            raise ComponentNotFoundException(
                f"Function {self._function_name} does not exist in file {self._filepath}"
            )
        app_fn = namespace[self._function_name]
        fn_desc, _ = get_fn_docstring(app_fn)
        return [
            _Component(
                name=f"{self._filepath}:{self._function_name}",
                description=fn_desc,
                fn_name=self._function_name,
                fn=app_fn,
                validation_errors=validation_errors,
            )
        ]


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


def _find_components() -> Dict[str, _Component]:
    global _components
    if not _components:
        _components = _load_components()
    return none_throws(_components)


def _is_custom_component(component_name: str) -> bool:
    return ":" in component_name


def _find_custom_components(name: str) -> Dict[str, _Component]:
    if ":" not in name:
        raise ValueError(
            f"Invalid custom component: {name}, valid template : `FILEPATH`:`FUNCTION_NAME`"
        )
    filepath, component_name = name.split(":")
    components = CustomComponentsFinder(filepath, component_name).find()
    return {component.name: component for component in components}


def get_components() -> Dict[str, _Component]:
    """
    Returns all builtint components and components registered
    in the ``[torchx.components] entry_points.txt``
    Each line is a key, value pair in a format:
    ::
      foo = test.bar

    Where ``test.bar`` is a valid path to the python module and ``foo`` is alias.
    Note: the module mast be discoverable by the torchx.

    Returns:
        Components in a format : {ALIAS: LIST_OF_COMPONENTS}
    """
    valid_components: Dict[str, _Component] = {}
    for component_name, component in _find_components().items():
        if len(component.validation_errors) == 0:
            valid_components[component_name] = component
    return valid_components


def get_component(name: str) -> _Component:
    """
    Retrieves components by the provided name.

    Returns:
        Component or None if no component with ``name`` exists
    """
    if _is_custom_component(name):
        components = _find_custom_components(name)
    else:
        components = _find_components()
    if name not in components:
        raise ComponentNotFoundException(
            f"Component `{name}` not found. Please make sure it is one of the "
            "builtins: `torchx builtins`. Or registered via `[torchx.components]` "
            "entry point (see: https://pytorch.org/torchx/latest/configure.html)"
        )

    component = components[name]
    if len(component.validation_errors) > 0:
        validation_msg = "\n".join(component.validation_errors)
        raise ComponentValidationException(
            f"Component {name} has validation errors: \n {validation_msg}"
        )
    return component


def get_builtin_source(name: str) -> str:
    """
    Returns a string of the the builtin component's function source code
    with all the import statements. Intended to be used to make a copy
    of the builtin component to use as a template for further customization.

    For simplicity import statements are read literally from the python file
    where the builtin component is defined. All lines that start with
    "import " and "from " preceding the function declaration
    (e.g. ``def builtin_name(...):`` are considered necessary import statements
    and hence included in the returned string.

    Therefore, it is possible to get additional unused import statements,
    which can happen if multiple builtins are defined in the same file.
    Make sure to pass the copy through a linter so that import statements
    are optimized and formatting adheres to your organization's standards.
    """

    component = get_component(name)
    fn = component.fn
    fn_name = component.name.split(".")[-1]

    # grab only the literal import statements BEFORE the builtin function def
    with open(inspect.getfile(component.fn), "r") as f:
        import_stmts = []
        for line in f.readlines():
            if line.startswith("import ") or line.startswith("from "):
                import_stmts.append(line.rstrip("\n"))
            elif line.startswith(f"def {fn_name}("):
                break

    fn_src = inspect.getsource(fn)

    return "\n".join([*import_stmts, "\n", fn_src, "\n"])
