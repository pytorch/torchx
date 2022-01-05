#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import sys
from typing import List


COPYRIGHT_NOTICE: str = """
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
""".strip()


def main(files: List[str]) -> None:
    missing = []
    for path in files:
        with open(path, "rt") as f:
            data = f.read()

        data = data.strip()
        # don't lint empty files
        if len(data) == 0:
            continue

        # skip the interpreter command and formatting
        while data.startswith("#!") or data.startswith("# -*-"):
            data = data[data.index("\n") + 1 :]

        if not data.startswith(COPYRIGHT_NOTICE):
            missing.append(path)

    if len(missing) > 0:
        print(f"{COPYRIGHT_NOTICE}\n")
        print("Please add the copyright notice to all listed files:\n")
        for path in missing:
            print(path)
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])
