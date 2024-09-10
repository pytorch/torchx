#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# pyre-strict

import uuid
from typing import Optional

CURRENT_SESSION_ID: Optional[str] = None


def get_session_id_or_create_new() -> str:
    """
    Returns the current session ID, or creates a new one if none exists.
    The session ID remains the same as long as it is in the same process.
    """
    global CURRENT_SESSION_ID
    if CURRENT_SESSION_ID:
        return CURRENT_SESSION_ID
    session_id = str(uuid.uuid4())
    CURRENT_SESSION_ID = session_id
    return session_id
