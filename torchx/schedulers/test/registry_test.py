#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from typing import Any, Dict, Optional
from unittest.mock import MagicMock, patch

from torchx.schedulers import get_schedulers
from torchx.schedulers.local_scheduler import LocalScheduler


class spy_load_group:
    def __call__(
        self,
        group: str,
        default: Dict[str, Any],
        ignore_missing: Optional[bool] = False,
    ) -> Dict[str, Any]:
        return default


class SchedulersTest(unittest.TestCase):
    @patch("torchx.schedulers.load_group", new_callable=spy_load_group)
    def test_get_local_schedulers(self, mock_load_group: MagicMock) -> None:
        schedulers = get_schedulers(session_name="test_session")
        self.assertTrue(isinstance(schedulers["local"], LocalScheduler))
        self.assertTrue(isinstance(schedulers["default"], LocalScheduler))

        self.assertEqual("test_session", schedulers["local"].session_name)
        self.assertEqual("test_session", schedulers["default"].session_name)
