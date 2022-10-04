#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from collections import defaultdict
from typing import cast, DefaultDict, Dict, Iterable, Mapping, Optional, Tuple
from unittest import mock, TestCase
from unittest.mock import patch

import torchx.tracker.api as api
from torchx.tracker.api import (
    AppRun,
    Lineage,
    tracker_config_env_var_name,
    TRACKER_ENV_VAR_NAME,
    TrackerArtifact,
    TrackerBase,
    TrackerSource,
)

RunId = str

DEFAULT_SOURCE: str = "__parent__"


class TestTrackerBackend(api.TrackerBase):
    def __init__(self, config_path: Optional[str] = None) -> None:
        self._artifacts: DefaultDict[
            RunId,
            DefaultDict[str, Tuple[str, Optional[Mapping[str, object]]]],
        ] = defaultdict(lambda: defaultdict())
        self._metdata: DefaultDict[RunId, DefaultDict[str, object]] = defaultdict(
            lambda: defaultdict()
        )

        self._sources: DefaultDict[RunId, Dict[str, str]] = defaultdict(dict)
        self.config_path = config_path

    def add_artifact(
        self,
        run_id: str,
        name: str,
        path: str,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        self._artifacts[run_id][name] = (path, metadata)

    def artifacts(self, run_id: str) -> Mapping[str, TrackerArtifact]:
        exp_artifacts = self._artifacts[run_id]
        return {
            k: TrackerArtifact(k, path, metadata)
            for k, [path, metadata] in exp_artifacts.items()
        }

    def add_metadata(self, run_id: str, **kwargs: object) -> None:
        self._metdata[run_id].update(kwargs)

    def metadata(self, run_id: str) -> Mapping[str, object]:
        return self._metdata[run_id]

    def add_source(
        self,
        run_id: str,
        source_id: str,
        artifact_name: Optional[str],
    ) -> None:
        if not artifact_name:
            artifact_name = DEFAULT_SOURCE
        self._sources[run_id][artifact_name] = source_id

    def sources(
        self,
        run_id: str,
        artifact_name: Optional[str] = None,
    ) -> Iterable[TrackerSource]:
        source_data = self._sources[run_id]

        sources = []
        for artifact_name, source_id in self._sources[run_id].items():
            if artifact_name == DEFAULT_SOURCE:
                sources.append(TrackerSource(source_id, None))
            else:
                sources.append(TrackerSource(source_id, artifact_name))
        return sources

    def lineage(self, run_id: str) -> Lineage:
        return Lineage()

    def run_ids(self, **kwargs: str) -> Iterable[str]:
        return []


class AppRunApiTest(TestCase):
    def setUp(self) -> None:
        os.environ["TORCHX_JOB_ID"] = "scheduler://sesison/app_id"
        self.tracker = TestTrackerBackend()
        self.run_id = "run_id"
        self.model_run = AppRun(self.run_id, [self.tracker])

    @mock.patch.dict(
        os.environ, {TRACKER_ENV_VAR_NAME: "tracker1", "TORCHX_JOB_ID": "run_id"}
    )
    def test_env_factory_test(self) -> None:
        with patch(
            "torchx.tracker.api.load_group",
            return_value={"tracker1": tracker_factory},
        ):
            app_run = api.AppRun.from_env()
            self.assertEqual(app_run.run_id, self.run_id)
            self.assertEqual(TestTrackerBackend, type(app_run.backends[0]))

    def test_get_exierment_run_id(self) -> None:
        self.assertEqual(self.run_id, self.model_run.job_id())

    def test_add_artifact(self) -> None:
        self.model_run.add_artifact("output", "/directory")
        artifacts = self.tracker.artifacts(self.run_id)
        self.assertEqual("/directory", artifacts["output"].path)
        self.assertIsNone(artifacts["output"].metadata)

    def test_add_artifact_with_metadata(self) -> None:
        self.model_run.add_artifact("output", "/directory", {"date": "2020/01/01"})
        artifacts = self.tracker.artifacts(self.run_id)
        metadata = artifacts["output"].metadata
        self.assertIsNotNone(metadata)
        if metadata:
            self.assertEqual("2020/01/01", metadata["date"])

    def test_add_source(self) -> None:
        self.model_run.add_source("parent_model_run")
        sources = self.model_run.sources()
        self.assertEqual(1, len(list(sources)))
        source = list(sources)[0]
        self.assertEqual(source.parent.run_id, "parent_model_run")
        self.assertEqual(source.parent.backends, self.model_run.backends)

    def test_add_metadata(self) -> None:
        self.model_run.add_metadata(lr=0.01, bs=25)

        metadata = self.tracker.metadata(self.run_id)
        self.assertEqual(metadata, {"lr": 0.01, "bs": 25})

    def test_add_metadata_should_update_values(self) -> None:
        self.model_run.add_metadata(lr=0.01, bs=25)
        self.model_run.add_metadata(bs=10)

        metadata = self.tracker.metadata(self.run_id)
        self.assertEqual(metadata, {"lr": 0.01, "bs": 10})

    def test_lineage(self) -> None:
        self.tracker.lineage(self.run_id)

    def test_run_ids(self) -> None:
        self.tracker.run_ids()


def tracker_factory(config: Optional[str] = None) -> TrackerBase:
    return TestTrackerBackend(config)


class TrackerFromEnvironTest(TestCase):
    def test_config_env_var_name(self) -> None:
        value = tracker_config_env_var_name("tracker1")
        self.assertEqual(value, "TORCHX_TRACKER_TRACKER1_CONFIG")

    def test_trackers_from_environ_with_no_trackers_specified(self) -> None:
        self.assertEqual(len(list(api.trackers_from_environ())), 0)

    @mock.patch.dict(os.environ, {TRACKER_ENV_VAR_NAME: "tracker1"})
    def test_tracker_from_environ(self) -> None:
        with patch(
            "torchx.tracker.api.load_group",
            return_value={"tracker1": tracker_factory},
        ):
            trackers = api.trackers_from_environ()
            self.assertEqual(1, len(list(trackers)))
            self.assertEqual(TestTrackerBackend, type(trackers[0]))

    @mock.patch.dict(
        os.environ,
        {
            TRACKER_ENV_VAR_NAME: "tracker1",
            tracker_config_env_var_name("tracker1"): "myconfig.txt",
        },
    )
    def test_tracker_from_environ_with_config_setting(self) -> None:
        with patch(
            "torchx.tracker.api.load_group",
            return_value={"tracker1": tracker_factory},
        ):
            trackers = api.trackers_from_environ()
            tracker = cast(TestTrackerBackend, list(trackers)[0])
            self.assertEqual("myconfig.txt", tracker.config_path)

    def test_trackerfrom_environ_that_wasnt_setup(self) -> None:
        trackers = api.trackers_from_environ()
        self.assertEqual(len(list(trackers)), 0)

    @mock.patch.dict(os.environ, {TRACKER_ENV_VAR_NAME: "tracker1"})
    def test_tracker_from_environ_with_missing_entrypoint(self) -> None:
        with patch(
            "torchx.tracker.api.load_group",
            return_value={},
        ):
            trackers = api.trackers_from_environ()
            self.assertEqual(0, len(list(trackers)))
