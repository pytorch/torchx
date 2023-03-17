#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import unittest
from unittest.mock import Mock, patch

from torchx.cli.cmd_tracker import CmdTracker
from torchx.tracker.api import TrackerArtifact


class CmdTrackerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tracker_name = "my_tracker"
        self.tracker_config = "myconfig.txt"
        self.tracker = Mock()
        self.test_run_id = "scheduler://session/app_id"
        with patch("torchx.cli.cmd_tracker.get_configured_trackers") as config_fn:
            config_fn.return_value = {self.tracker_name: self.tracker_config}
            with patch("torchx.cli.cmd_tracker.build_trackers") as tracker_fn:
                tracker_fn.return_value = [self.tracker]
                self.cmd = CmdTracker()
        self.parser = argparse.ArgumentParser()
        self.cmd.add_arguments(self.parser)

    def test_initalize_using_config(self) -> None:
        self.assertEqual(self.cmd.tracker, self.tracker)

    def test_initalize_on_missing_tracker(self) -> None:
        with patch("torchx.cli.cmd_tracker.build_trackers") as tracker_fn:
            tracker_fn.return_value = []
            CmdTracker()

    def test_list_jobs_cmd(self) -> None:
        self.tracker.run_ids.return_value = ["id1", "id2"]
        args = self.parser.parse_args(["list", "jobs"])
        args.func(args)
        self.tracker.run_ids.assert_called_once()

    def test_list_jobs_cmd_with_missing_tracker(self) -> None:
        with patch("torchx.cli.cmd_tracker.get_configured_trackers") as config_fn:
            config_fn.return_value = {}
            cmd = CmdTracker()
            parser = argparse.ArgumentParser()
            cmd.add_arguments(parser)
            args = parser.parse_args(["list", "jobs"])
            with self.assertRaises(SystemExit):
                args.func(args)

    def test_list_jobs_cmd_with_parent_id(self) -> None:
        expected_parent_run_id = "my_experiment"
        self.tracker.run_ids.return_value = ["id1", "id2"]
        args = self.parser.parse_args(
            ["list", "jobs", "--parent-run-id", expected_parent_run_id]
        )
        args.func(args)
        self.tracker.run_ids.assert_called_once_with(
            parent_run_id=expected_parent_run_id
        )

    def test_list_metadata_cmd(self) -> None:
        self.tracker.metadata.return_value = {"v1": 1, "v2": "2"}

        args = self.parser.parse_args(["list", "metadata", self.test_run_id])
        args.func(args)
        self.tracker.metadata.assert_called_once()

    def test_list_artifacts_cmd(self) -> None:
        self.tracker.artifacts.return_value = {
            "test_artifact": TrackerArtifact("test_artifact", "/path", None)
        }

        args = self.parser.parse_args(["list", "artifacts", self.test_run_id])
        args.func(args)
        self.tracker.artifacts.assert_called_once_with(self.test_run_id)
