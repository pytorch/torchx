# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from abc import ABC, abstractmethod
from typing import Callable

from torchx.specs import AppDef, SchedulerBackend
import torchx.components.dist as dist_components
import torchx.components.serve as serve_components
import torchx.components.utils as utils_components
from torchx.components.component_test_base import ComponentUtils


class ComponentProvider(ABC):
    def __init__(self, scheduler: SchedulerBackend, image: str) -> None:
        self._scheduler = scheduler
        self._image = image

    @abstractmethod
    def get_app_def(self) -> AppDef:
        pass

    def post_component_exec(self) -> None:
        pass

    def validate_linter_errors(self, component_fn: Callable[..., AppDef]) -> None:
        linter_errors = ComponentUtils.get_linter_errors_for_component(component_fn)
        if len(linter_errors) > 0:
            errors = "\n".join([error.description for error in linter_errors])
            raise AssertionError(f"Component {component_fn.__name__} contains linter errors: \n {errors}")


class DDPComponentProvider(ComponentProvider):
    def get_app_def(self) -> AppDef:
        self.validate_linter_errors(dist_components.ddp)
        args = ["torchx.apps.test.dummy_script"]
        rdzv_backend: str = "c10d"
        rdzv_endpoint: str = "localhost:29400"
        return dist_components.ddp(
            *args,
            entrypoint="-m",
            image=self._image,
            rdzv_backend=rdzv_backend,
            rdzv_endpoint=rdzv_endpoint
        )


class ServeComponentProvider(ComponentProvider):
    def get_app_def(self) -> AppDef:
        self.validate_linter_errors(serve_components.torchserve)
        return serve_components.torchserve(
            model_path="",
            management_api="",
            image=self._image,
            dryrun=True,
        )


class BoothComponentProvider(ComponentProvider):
    def get_app_def(self) -> AppDef:
        self.validate_linter_errors(utils_components.booth)
        return utils_components.booth(
            x1=1.0,
            x2=2.0,
            image=self._image,
        )


class ShComponentProvider(ComponentProvider):
    def get_app_def(self) -> AppDef:
        self.validate_linter_errors(utils_components.sh)
        return utils_components.sh(
            *["echo", "test"],
            image=self._image,
        )


class TouchComponentProvider(ComponentProvider):
    def __init__(self, image: str, scheduler: SchedulerBackend):
        super(TouchComponentProvider, self).__init__(image, scheduler)
        if self._scheduler == "local":
            self._file_path = os.path.join(os.getcwd(), "test.txt")
        else:
            self._file_path = "test.txt"

    def get_app_def(self) -> AppDef:
        self.validate_linter_errors(utils_components.touch)
        return utils_components.touch(
            file="test.txt",
            image=self._image,
        )

    def post_component_exec(self) -> None:
        if self._scheduler == "local" and os.path.exists(self._file_path):
            os.remove(self._file_path)


class CopyComponentProvider(ComponentProvider):
    def __init__(self, image: str, scheduler: SchedulerBackend):
        super(CopyComponentProvider, self).__init__(image, scheduler)
        if self._scheduler == "local":
            self._src_path = os.path.join(os.getcwd(), "copy_test.txt")
            self._dst_path = os.path.join(os.getcwd(), "copy_test.txt.copy")
            self._process_local_sched()
        else:
            self._src_path = "README.md"
            self._dst_path = "README.md.copy"

    def _process_local_sched(self) -> None:
        if not os.path.exists(self._src_path):
            with open(self._src_path, "w") as f:
                f.write("test data")

    def post_component_exec(self) -> None:
        if self._scheduler == "local" and os.path.exists(self._dst_path):
            os.remove(self._dst_path)

    def get_app_def(self) -> AppDef:
        self.validate_linter_errors(utils_components.copy)
        return utils_components.copy(
            src=self._src_path,
            dst=self._dst_path,
            image=self._image
        )
