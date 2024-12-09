#!/bin/sh
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -eux

## HACK: run pyre binary manually to see the error

echo '{
  "source_paths": {
    "kind": "simple",
    "paths": [
      "/home/runner/work/torchx/torchx/scripts",
      "/home/runner/work/torchx/torchx"
    ]
  },
  "search_paths": [
    "/home/runner/work/torchx/torchx/stubs",
    "/opt/hostedtoolcache/Python/3.10.15/x64/lib/python3.10/site-packages",
    "/opt/hostedtoolcache/Python/3.10.15/x64/lib/pyre_check/typeshed/stdlib",
    "/opt/hostedtoolcache/Python/3.10.15/x64/lib/pyre_check/typeshed/stubs/ExifRead",
    "/opt/hostedtoolcache/Python/3.10.15/x64/lib/pyre_check/typeshed/stubs/PyMySQL",
    "/opt/hostedtoolcache/Python/3.10.15/x64/lib/pyre_check/typeshed/stubs/PyYAML",
    "/opt/hostedtoolcache/Python/3.10.15/x64/lib/pyre_check/typeshed/stubs/aiofiles",
    "/opt/hostedtoolcache/Python/3.10.15/x64/lib/pyre_check/typeshed/stubs/chevron",
    "/opt/hostedtoolcache/Python/3.10.15/x64/lib/pyre_check/typeshed/stubs/colorama",
    "/opt/hostedtoolcache/Python/3.10.15/x64/lib/pyre_check/typeshed/stubs/ldap3",
    "/opt/hostedtoolcache/Python/3.10.15/x64/lib/pyre_check/typeshed/stubs/mysqlclient",
    "/opt/hostedtoolcache/Python/3.10.15/x64/lib/pyre_check/typeshed/stubs/paramiko",
    "/opt/hostedtoolcache/Python/3.10.15/x64/lib/pyre_check/typeshed/stubs/psycopg2",
    "/opt/hostedtoolcache/Python/3.10.15/x64/lib/pyre_check/typeshed/stubs/pycurl",
    "/opt/hostedtoolcache/Python/3.10.15/x64/lib/pyre_check/typeshed/stubs/python-dateutil",
    "/opt/hostedtoolcache/Python/3.10.15/x64/lib/pyre_check/typeshed/stubs/pytz",
    "/opt/hostedtoolcache/Python/3.10.15/x64/lib/pyre_check/typeshed/stubs/regex",
    "/opt/hostedtoolcache/Python/3.10.15/x64/lib/pyre_check/typeshed/stubs/requests",
    "/opt/hostedtoolcache/Python/3.10.15/x64/lib/pyre_check/typeshed/stubs/retry",
    "/opt/hostedtoolcache/Python/3.10.15/x64/lib/pyre_check/typeshed/stubs/tqdm",
    "/opt/hostedtoolcache/Python/3.10.15/x64/lib/pyre_check/typeshed/stubs/ujson"
  ],
  "excludes": [
    ".*/build/.*",
    ".*/docs/.*",
    ".*/setup.py",
    ".*/IPython/core/tests/nonascii.*",
    ".*/torchx/examples/apps/compute_world_size/.*"
  ],
  "checked_directory_allowlist": [
    "/home/runner/work/torchx/torchx",
    "/home/runner/work/torchx/torchx/scripts"
  ],
  "checked_directory_blocklist": [
    "/home/runner/work/torchx/torchx/stubs"
  ],
  "extensions": [],
  "log_path": "/home/runner/work/torchx/torchx/.pyre",
  "global_root": "/home/runner/work/torchx/torchx",
  "debug": false,
  "python_version": {
    "major": 3,
    "minor": 10,
    "micro": 15
  },
  "system_platform": "linux",
  "shared_memory": {},
  "parallel": true,
  "number_of_workers": 1,
  "additional_logging_sections": [
    "-progress"
  ],
  "show_error_traces": false,
  "strict": true
}' >/tmp/pyre_arguments

/opt/hostedtoolcache/Python/3.10.15/x64/bin/pyre.bin check /tmp/pyre_arguments

pyre --version
pyre --noninteractive check
