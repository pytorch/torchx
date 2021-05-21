#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import unittest
from unittest.mock import patch, MagicMock

from torchx.runner.events import (
    _get_or_create_logger,
    SourceType,
    TorchxEvent,
    log_event,
)


class TorchxEventLibTest(unittest.TestCase):
    def assert_event(
        self, actual_event: TorchxEvent, expected_event: TorchxEvent
    ) -> None:
        self.assertEqual(actual_event.session, expected_event.session)
        self.assertEqual(actual_event.scheduler, expected_event.scheduler)
        self.assertEqual(actual_event.api, expected_event.api)
        self.assertEqual(actual_event.app_id, expected_event.app_id)
        self.assertEqual(actual_event.runcfg, expected_event.runcfg)
        self.assertEqual(actual_event.source, expected_event.source)

    @patch("torchx.runner.events.get_logging_handler")
    def test_get_or_create_logger(self, logging_handler_mock: MagicMock) -> None:
        logging_handler_mock.return_value = logging.NullHandler()
        logger = _get_or_create_logger("test_destination")
        self.assertIsNotNone(logger)
        self.assertEqual(1, len(logger.handlers))
        self.assertIsInstance(logger.handlers[0], logging.NullHandler)

    def test_event_created(self) -> None:
        event = TorchxEvent(
            session="test_session", scheduler="test_scheduler", api="test_api"
        )
        self.assertEqual("test_session", event.session)
        self.assertEqual("test_scheduler", event.scheduler)
        self.assertEqual("test_api", event.api)
        self.assertEqual(SourceType.UNKNOWN, event.source)

    def test_event_deser(self) -> None:
        event = TorchxEvent(
            session="test_session",
            scheduler="test_scheduler",
            api="test_api",
            source=SourceType.EXTERNAL,
        )
        json_event = event.serialize()
        deser_event = TorchxEvent.deserialize(json_event)
        self.assert_event(event, deser_event)


@patch("torchx.runner.events.record")
class LogEventTest(unittest.TestCase):
    def assert_torchx_event(self, expected: TorchxEvent, actual: TorchxEvent) -> None:
        self.assertEqual(expected.session, actual.session)
        self.assertEqual(expected.app_id, actual.app_id)
        self.assertEqual(expected.api, actual.api)
        self.assertEqual(expected.source, actual.source)

    def test_create_context(self, _) -> None:
        cfg = json.dumps({"test_key": "test_value"})
        context = log_event("test_call", "local", "test_app_id", cfg)
        expected_torchx_event = TorchxEvent(
            "test_app_id", "local", "test_call", "test_app_id", cfg
        )
        self.assert_torchx_event(expected_torchx_event, context._torchx_event)

    def test_record_event(self, record_mock: MagicMock) -> None:
        cfg = json.dumps({"test_key": "test_value"})
        expected_torchx_event = TorchxEvent(
            "test_app_id", "local", "test_call", "test_app_id", cfg
        )
        with log_event("test_call", "local", "test_app_id", cfg) as ctx:
            pass
        self.assert_torchx_event(expected_torchx_event, ctx._torchx_event)

    def test_record_event_with_exception(self, record_mock: MagicMock) -> None:
        cfg = json.dumps({"test_key": "test_value"})
        with self.assertRaises(RuntimeError):
            with log_event("test_call", "local", "test_app_id", cfg) as ctx:
                raise RuntimeError("test error")
        self.assertTrue("test error" in ctx._torchx_event.raw_exception)
