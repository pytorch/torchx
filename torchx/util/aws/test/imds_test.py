import socket
import unittest
from unittest import mock
from unittest.mock import MagicMock

from torchx.util.aws.imds import imds_fetch, instance_metadata, NULL

from requests.models import Response

REQUESTS_GET = "requests.get"
IMDS_FETCH = "torchx.util.aws.imds.imds_fetch"


def http_ok(text: str) -> Response:
    ENCODING = "UTF-8"
    resp = Response()
    resp.encoding = ENCODING
    resp._content = text.encode(ENCODING)
    resp.status_code = 200
    return resp


def http_error(code: int = 404, reason: str = "Not Found") -> Response:
    resp = Response()
    resp.status_code = code
    resp.reason = reason
    return resp


class IMDSTest(unittest.TestCase):
    def setUp(self) -> None:
        imds_fetch.cache_clear()
        instance_metadata.cache_clear()

    def tearDown(self) -> None:
        imds_fetch.cache_clear()
        instance_metadata.cache_clear()

    @mock.patch(REQUESTS_GET, return_value=http_ok("p4d.24xlarge"))
    def test_imds_fetch(self, _: MagicMock) -> None:
        self.assertEqual("p4d.24xlarge", imds_fetch("instance-type"))

    @mock.patch(REQUESTS_GET, return_value=http_error())
    def test_imds_not_available_default(self, _: MagicMock) -> None:
        self.assertIsNone(imds_fetch("instancetype"))

    @mock.patch(REQUESTS_GET, return_value=http_error())
    def test_instance_metadata_no_imds(self, _: MagicMock) -> None:
        self.assertEqual("localhost", instance_metadata().instance_id)
        self.assertEqual(NULL, instance_metadata().instance_type)
        self.assertEqual(NULL, instance_metadata().availability_zone)
        self.assertEqual(NULL, instance_metadata().region)
        self.assertEqual(socket.gethostname(), instance_metadata().hostname)
        self.assertEqual(NULL, instance_metadata().local_hostname)
        self.assertEqual(NULL, instance_metadata().public_hostname)
        self.assertEqual(NULL, instance_metadata().ami_id)

    def test_instance_metadata(self) -> None:
        def mock_imds_fetch(metadata: str = "") -> str:
            if metadata == "":
                return "ami-id\nami-launch-index\nservices/\nsystem"
            if metadata == "instance-id":
                return "i-1a2b3c"
            elif metadata == "instance-type":
                return "p4d.24xlarge"
            elif metadata == "placement/availability-zone":
                return "us-west-2a"
            elif metadata == "placement/region":
                return "us-west-2"
            elif metadata == "hostname":
                return "ip-...us-west-2.compute.internal"
            elif metadata == "public-hostname":
                return "ec2-...compute.amazonaws.com"
            elif metadata == "local-hostname":
                return "ip-...us-west-2.compute.internal"
            elif metadata == "ami-id":
                return "ami-1a2b3c"
            else:
                raise ValueError(f"No test value mapped for: {metadata}")

        with mock.patch(IMDS_FETCH, wraps=mock_imds_fetch):
            md = instance_metadata()

            self.assertEqual("i-1a2b3c", md.instance_id)
            self.assertEqual("p4d.24xlarge", md.instance_type)
            self.assertEqual("us-west-2a", md.availability_zone)
            self.assertEqual("us-west-2", md.region)
            self.assertEqual("ip-...us-west-2.compute.internal", md.hostname)
            self.assertEqual("ec2-...compute.amazonaws.com", md.public_hostname)
            self.assertEqual("ip-...us-west-2.compute.internal", md.local_hostname)
            self.assertEqual("ami-1a2b3c", md.ami_id)
