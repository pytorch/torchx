#!/usr/bin/env python3
import os
from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock

from torchx.util.aws.region import get_region
from torchx.test.fixtures import TestWithTmpDir

INSTANCE_METADATA = "torchx.util.aws.imds.instance_metadata"


class Boto3UtilTest(TestWithTmpDir):
    @mock.patch(INSTANCE_METADATA)
    def test_get_region(self, mock_instance_metadata: MagicMock) -> None:
        # write a region that we would never use as a devasg
        aws_config_file = self.write(
            str(Path(".aws") / "config"),
            [
                """
[default]
region=ap-east-1
"""
            ],
        )

        with mock.patch.dict(
            os.environ, {"AWS_CONFIG_FILE": str(aws_config_file)}, clear=True
        ):
            self.assertEqual("ap-east-1", get_region())
            mock_instance_metadata.assert_not_called()

    @mock.patch(INSTANCE_METADATA)
    def test_get_region_fallback_to_imds(
        self, mock_instance_metadata: MagicMock
    ) -> None:
        mock_instance_metadata.return_value.region = "ap-east-1"

        # point the config file to a file that does not exist
        # to force the fallback
        with mock.patch.dict(
            os.environ,
            {"AWS_CONFIG_FILE": str(self.tmpdir / "non-existent-config")},
            clear=True,
        ):
            self.assertEqual("ap-east-1", get_region())
            mock_instance_metadata.assert_called_once()
