# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping, Optional

from torchx.util.entrypoints import load_group

logger: logging.Logger = logging.getLogger(__name__)

TRACKER_ENV_VAR_NAME = "TORCHX_TRACKERS"
TRACKER_PARENT_RUN_ENV_VAR_NAME = "TORCHX_PARENT_RUN_ID"


@dataclass
class TrackerSource:
    """
    Dataclass to represent sources at backend tracker level

    Args:
        source_run_id(str): source ID that can be either other TorchX handle or ID of external entity such as experiment
        artifact_name(Optional[str]): type of of source. Can be interpreted as type of relationship that can be used for filtering.
    """

    source_run_id: str
    artifact_name: Optional[str]


@dataclass
class TrackerArtifact:
    """
    Dataclass to represent artifacts at backend tracker level

    Args:
        name(str): Name of the artifact
        path(str): Path to actual artifact
        metadata(Optional[Mapping[str, object]]): Additional metadata to store about artifact
    """

    name: str
    path: str
    metadata: Optional[Mapping[str, object]]


@dataclass
class AppRunTrackableSource:
    """
    Dataclass to represent sources at user API level

    Args:
        parent(AppRun): source AppRun that current run is derived from.
        artifact_name(Optional[str]): type of artifact represening parent AppRun.
    """

    parent: AppRun
    artifact_name: Optional[str]


class Lineage:
    ...


class TrackerBase(ABC):
    """
    Abstraction of tracking solution implementations/services.

    This API is stil experimental and may change in the future to a large extend.
    """

    @abstractmethod
    def add_artifact(
        self,
        run_id: str,
        name: str,
        path: str,
        metadata: Optional[Mapping[str, object]] = None,
    ) -> None:
        """
        Adds an artifact to the tracker with the specified name, path and any arbitrary metadata.
        """
        ...

    @abstractmethod
    def artifacts(self, run_id: str) -> Mapping[str, TrackerArtifact]:
        """
        Fetches all of the artifacts for the specified run or the current one if not specified.
        """
        ...

    @abstractmethod
    def add_metadata(self, run_id: str, **kwargs: object) -> None:
        """
        Adds any arbitrary metadata values to the specified experiment.
        """
        ...

    @abstractmethod
    def metadata(self, run_id: str) -> Mapping[str, object]:
        """
        Fetches the metadata for the specified experiment.
        """
        ...

    @abstractmethod
    def add_source(
        self,
        run_id: str,
        source_id: str,
        artifact_name: Optional[str] = None,
    ) -> None:
        """
        Adds a link to a different job identifying the lineage of the current experiment
        and the specific artifact that's being used.
        """
        ...

    @abstractmethod
    def sources(
        self,
        run_id: str,
        artifact_name: Optional[str] = None,
    ) -> Iterable[TrackerSource]:
        """
        Returns sources for the specified run. Artifact name can be used to filter the sources.
        """
        ...

    @abstractmethod
    def lineage(self, run_id: str) -> Lineage:
        """
        Returns the lineage for the specified experiment. The lineage includes all
        parent jobs and artifacts this job depends on as well as any jobs that are consuming
        artifacts from this job.
        """
        ...

    @abstractmethod
    def run_ids(self, **kwargs: str) -> Iterable[str]:
        """
        Returns list of experiment run ids. Optionally includes filter parameters.
        """
        ...


def tracker_config_env_var_name(entrypoint_key: str) -> str:
    """Utility method to derive tracker config env variable name given tracker name"""
    return f"TORCHX_TRACKER_{entrypoint_key.upper()}_CONFIG"


def _extract_tracker_name_and_config_from_environ() -> Mapping[str, Optional[str]]:
    if TRACKER_ENV_VAR_NAME not in os.environ:
        logger.info("No trackers were configured, skipping setup.")
        return {}

    tracker_backend_entrypoints = os.environ[TRACKER_ENV_VAR_NAME]
    logger.info(f"Trackers specified {tracker_backend_entrypoints}")

    entries = {}
    for entrypoint_key in tracker_backend_entrypoints.split(","):
        config = None
        config_env_name = tracker_config_env_var_name(entrypoint_key)
        if config_env_name in os.environ:
            config = os.environ[config_env_name]
        entries[entrypoint_key] = config

    return entries


def build_trackers(
    entrypoint_and_config: Mapping[str, Optional[str]]
) -> Iterable[TrackerBase]:
    trackers = []

    entrypoint_factories = load_group("torchx.tracker")
    if not entrypoint_factories:
        logger.warn(
            "No 'torchx.tracker' entry_points are defined. Tracking will not capture any data."
        )
        return trackers

    for entrypoint_key, config in entrypoint_and_config.items():
        logger.info(f"Configuring tracker {entrypoint_key}")
        if entrypoint_key not in entrypoint_factories:
            logger.warn(
                f"Coult not find '{entrypoint_key}' tracker entrypoint, skipping."
            )
            continue
        factory = entrypoint_factories[entrypoint_key]
        if config:
            logger.info(f"Trackers config specified for {entrypoint_key} as {config}")
            tracker = factory(config)
        else:
            logger.info(f"No trackers config specified for {entrypoint_key}")
            tracker = factory(None)
        trackers.append(tracker)
    return trackers


def trackers_from_environ() -> Iterable[TrackerBase]:
    """
    Builds list of Trackers that will be used to persist tracking information and will be used by AppRun
        to delegate calls to the each instance.
    Expects `TORCHX_TRACKERS` env variable to contain list of entry-point factory keys, separated by comma.
    Optionally, for each tracker key user can pass config string value under `TORCHX_TRACKER_<ENTRYPOINT_NAME>_CONFIG` env variable.
        It is up to each implementation to interpret the value, eg it can an encoded data or path to richer config properties file.

    Entry-points(factory methods) must exist in runtime when job is running since this runs within user-job space.
    """

    entrypoint_and_config = _extract_tracker_name_and_config_from_environ()
    if entrypoint_and_config:
        return build_trackers(entrypoint_and_config)
    return []


@dataclass
class AppRun:
    """
    Exposes tracker API to at the job level and should the only API that encapsulates that module implementation.

    This API is stil experimental and may change in the future.

    Args:
        id(str): identity of the job used by tracker API
        backends(Iterable[TrackerBase]): list of TrackerBase implementations that will be used to persist the data.
    """

    id: str
    backends: Iterable[TrackerBase]

    @staticmethod
    @lru_cache(maxsize=1)  # noqa: B019
    def run_from_env() -> AppRun:
        """Factory method do build AppRun that uses environment variabes to build it.

        Single instance of AppRun will be available for the duration of the application, thus expect that calling
        this method will retorn same AppRun and hence same instances of the backend implementations.
        """

        torchx_job_id = os.environ["TORCHX_JOB_ID"]

        trackers = trackers_from_environ()
        if TRACKER_PARENT_RUN_ENV_VAR_NAME in os.environ:
            parent_run_id = os.environ[TRACKER_PARENT_RUN_ENV_VAR_NAME]
            logger.info(f"Tracker parent run ID: '{parent_run_id}'")
            for tracker in trackers:
                tracker.add_source(torchx_job_id, parent_run_id, artifact_name=None)

        return AppRun(id=torchx_job_id, backends=trackers_from_environ())

    def add_metadata(self, **kwargs: object) -> None:
        """Stores metadata for the current run"""
        for backend in self.backends:
            backend.add_metadata(self.id, **kwargs)

    def add_artifact(
        self, name: str, path: str, metadata: Optional[Mapping[str, object]] = None
    ) -> None:
        """Stores artifacts for the current run

        Args:
            name(str): name of the artifact
            path(str): path of the artifact that is stored
            metadata(Optional[Mapping[str, object]]): optional metadata attached to artifact information
        """
        for backend in self.backends:
            backend.add_artifact(self.id, name, path, metadata)

    def job_id(self) -> str:
        """Current Id of the run"""
        return self.id

    def add_source(self, source_id: str, artifact_name: Optional[str] = None) -> None:
        """
        Attaches source to this run. Sources can be either other TorchX runs or external entities such as experiments that may
        or may not be queriable.

        Args:
            source_id(str): identity of the source
            artifact_name(Optional[str]): optional value of type of source
        """
        for backend in self.backends:
            backend.add_source(self.id, source_id, artifact_name)

    def sources(self) -> Iterable[AppRunTrackableSource]:
        """
        Returns `AppRunTrackableSource` for the run.

        Uses first backend to query this information, although it supports list of trackers in order
        to persist tracking information.
        """
        model_run_sources = []
        if self.backends:
            backend = next(iter(self.backends))
            sources = backend.sources(self.id)
            for source in sources:
                parent = AppRun(source.source_run_id, backends=self.backends)
                model_run_source = AppRunTrackableSource(parent, source.artifact_name)
                model_run_sources.append(model_run_source)

        return model_run_sources

    def children(self) -> Iterable[AppRun]:
        ...
