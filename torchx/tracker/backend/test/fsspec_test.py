#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest

import fsspec

from torchx.tracker.backend.fsspec import (
    _decode_torchx_run_id,
    _encode_torchx_run_id,
    create,
    FsspecTracker,
    generate_filename,
)


class FsspecTest(unittest.TestCase):
    def setUp(self) -> None:
        self.run_id = "local://session/id1"
        self.parent_run_id = "local://session/id0"

    def test_generate_filename(self) -> None:
        filename = generate_filename()
        self.assertIsNotNone(filename)

    def test_encode_decode_torchx_run(self) -> None:
        expected_run_id = "mast://session_id/app_id"

        self.assertEqual(
            _decode_torchx_run_id(_encode_torchx_run_id(expected_run_id)),
            expected_run_id,
        )

    def test_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as root_dir:
            # TODO use memory based fsspec, `exists` check on dirs are always falsy
            fs = fsspec.filesystem("file")
            tracker = FsspecTracker(fs, root_dir)
            tracker.add_artifact(self.run_id, "tf", "~/logs")

            self.assertEqual(tracker.artifacts(self.run_id)["tf"].path, "~/logs")

    def test_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as root_dir:
            fs = fsspec.filesystem("file")
            tracker = FsspecTracker(fs, root_dir)
            tracker.add_metadata(self.run_id, k1="1", k2=2)

            self.assertEqual(tracker.metadata(self.run_id)["k1"], "1")
            self.assertEqual(tracker.metadata(self.run_id)["k2"], 2)

    def test_sources(self) -> None:
        with tempfile.TemporaryDirectory() as root_dir:
            fs = fsspec.filesystem("file")
            tracker = FsspecTracker(fs, root_dir)
            tracker.add_source(self.run_id, self.parent_run_id)

            sources = list(tracker.sources(self.run_id))
            self.assertEqual(len(sources), 1)
            self.assertEqual(sources[0].source_run_id, self.parent_run_id)

    def test_sources_with_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as root_dir:
            fs = fsspec.filesystem("file")
            tracker = FsspecTracker(fs, root_dir)
            tracker.add_source(self.run_id, self.parent_run_id)
            parent_id_for_artifact = "local://session/parent"
            artifact = "some_propery_x"
            tracker.add_source(self.run_id, self.parent_run_id)
            tracker.add_source(self.run_id, parent_id_for_artifact, artifact)

            self.assertEqual(len(list(tracker.sources(self.run_id))), 2)

            sources = list(tracker.sources(self.run_id, artifact_name=artifact))
            self.assertEqual(len(sources), 1)
            self.assertEqual(
                sources[0].source_run_id,
                parent_id_for_artifact,
            )

    def test_run_ids(self) -> None:
        with tempfile.TemporaryDirectory() as root_dir:
            fs = fsspec.filesystem("file")
            tracker = FsspecTracker(fs, root_dir)
            tracker.add_artifact(self.parent_run_id, "tf", "path1")
            tracker.add_artifact(self.run_id, "tf", "path2", {"k": "v"})
            tracker.add_source(self.run_id, self.parent_run_id)

            run_ids = tracker.run_ids()
            self.assertEqual(set(run_ids), set([self.parent_run_id, self.run_id]))

    def test_run_ids_filter_by_parent(self) -> None:
        with tempfile.TemporaryDirectory() as root_dir:
            fs = fsspec.filesystem("file")
            tracker = FsspecTracker(fs, root_dir)
            tracker.add_artifact(self.parent_run_id, "tf", "path1")
            tracker.add_artifact(self.run_id, "tf", "path2", {"k": "v"})
            tracker.add_source(self.run_id, self.parent_run_id)

            run_ids = tracker.run_ids(parent_run_id=self.parent_run_id)
            self.assertEqual(run_ids, [self.run_id])

    def test_create(self) -> None:
        with tempfile.TemporaryDirectory() as root_dir:
            fs = fsspec.filesystem("file")
            tracker_root_path = f"{root_dir}/project"
            fs.mkdirs(tracker_root_path)
            with open(f"{root_dir}/config", mode="w") as config_file:
                config_file.write("protocol=file\n")
                config_file.write(f"root_path={tracker_root_path}\n")

            tracker = create(f"file://{root_dir}/config")
            self.assertEqual(
                tracker._path_builder.root_dir, tracker_root_path  # pyre-ignore
            )
