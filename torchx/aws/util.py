#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
import time
from typing import Iterator


def wait_for(
    msg: str, timeout: int = 300, interval: int = 1, print_spinner: bool = True
) -> Iterator[None]:
    """
    for _ in wait_for("asg to provision", timeout_sec, interval_sec):
        if check_condition():.
            break
    """
    spin = ["-", "/", "|", "\\", "-", "/", "|", "\\"]
    idx = 0
    start = time.time()
    max_time = start + timeout
    while True:
        if print_spinner:
            elapsed = time.time() - start
            print(
                f"Waiting for {msg}"
                f" ({elapsed:03.0f}/{timeout:3.0f}s elapsed) {spin[idx]}\r",
                end="",
            )
            sys.stdout.flush()
            idx = (idx + 1) % len(spin)

        if time.time() >= max_time:
            raise RuntimeError(f"Timed out while waiting for: {msg}")
        else:
            time.sleep(interval)
            yield
