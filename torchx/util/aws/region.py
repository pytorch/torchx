import boto3

from torchx.util.aws.imds import instance_metadata


def get_region() -> str:
    """
    Returns the region name (e.g. us-west-2) by:

    #. First see if the region can be obtained from the session chain
       e.g. if there is an ~/.aws/config or AWS_DEFAULT_REGION env var is set
    #. Otherwise, get the region from instance metadata (if running on EC2)


    We first use the session chain so that we can override the region using
    env vars or if we are running outside of an EC2 instance (e.g. laptop)
    """
    region = boto3.Session().region_name
    if not region:
        region = instance_metadata().region
    return region
