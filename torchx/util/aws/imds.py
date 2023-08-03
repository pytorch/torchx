"""
EC2 Instance Metadata Service related methods
"""
import logging
import socket
from dataclasses import dataclass
from functools import cache
from typing import Optional

import requests

from requests import HTTPError

IMDS_URL = "http://169.254.169.254/latest"
IMDS_METADATA_URL = f"{IMDS_URL}/meta-data"
NULL = "_NULL_"

log: logging.Logger = logging.getLogger(__name__)


@dataclass
class InstanceMetadata:
    """
    Struct representing the instance metadata as fetched from IMDS on the current host.
    Use :py:func:`ape.aws.ec2.imds.instance_metadata` to get a filled-out
    instance of this object.
    """

    instance_id: str
    instance_type: str
    hostname: str
    public_hostname: str
    local_hostname: str
    availability_zone: str
    region: str
    ami_id: str


@cache
def imds_fetch(metadata: str = "") -> Optional[str]:
    """
    Fetches the specified ``metadata`` from EC2's Instance MetaData Service (IMDS) running
    on the current host, by sending an HTTP GET request to ``http://169.254.169.254/latest/meta-data/<metadata>``.

    To get a list of all valid values of ``metadata`` run this method with no arguments then split
    the return value by new-line.


    Arguments:
        metadata: Name of the instance metadata to query (e.g. ``instance-type``)

    Returns:
        the specified ``metadata`` or ``None`` if IMDS cannot be reached

    """
    try:
        response = requests.get(f"{IMDS_METADATA_URL}/{metadata}")
    except Exception as e:
        log.warning(
            "Error querying IMDS instance metadata won't be available", exc_info=e
        )
        return None

    if response.ok:
        return response.text
    else:  # response NOT ok
        try:
            response.raise_for_status()
        except HTTPError:
            return None

    assert False, "Unreachable code!"  # pragma: no cover


@cache
def instance_metadata() -> InstanceMetadata:
    """
    Fetches the instance metadata for the current host from EC2's Instance Metadata Service (IMDS),
    which typically runs on localhost at ``http://169.254.169.254``.
    If IMDS cannot be reached for any reason returns an instance of :py:class:`InstanceMetadata`
    where all the fields are empty strings.

    .. note::
        This method is memoized (value is cached) hence, only the first call
        will actually hit IMDS, and subsequent calls will return the memoized
        value. Therefore, it is ok to call this function multiple times.

    """

    return InstanceMetadata(
        instance_id=imds_fetch("instance-id") or "localhost",
        instance_type=imds_fetch("instance-type") or NULL,
        availability_zone=imds_fetch("placement/availability-zone") or NULL,
        region=imds_fetch("placement/region") or NULL,
        hostname=imds_fetch("hostname") or socket.gethostname(),
        public_hostname=imds_fetch("public-hostname") or NULL,
        local_hostname=imds_fetch("local-hostname") or NULL,
        ami_id=imds_fetch("ami-id") or NULL,
    )
