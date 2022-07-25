#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from pydoc import locate

import yaml
from docutils import nodes
from sphinx.util.docutils import SphinxDirective


class RunOptsDirective(SphinxDirective):
    # this enables content in the directive
    has_content = True

    def run(self):
        raw_content = "\n".join(self.content)
        args = yaml.safe_load(raw_content)
        cls = locate(args["class"])
        scheduler = cls("docs")

        body = nodes.literal_block(text=str(scheduler.run_opts()))
        return [
            body,
        ]


def setup(app):
    app.add_directive("runopts", RunOptsDirective)

    return {
        "version": "0.1",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
