#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import getpass
import logging
import os
import random
import string
from typing import Dict, Tuple

import boto3
import botocore
from jinja2 import Template
from torchx.aws.util import wait_for


log: logging.Logger = logging.getLogger(__name__)


# TODO kiuk - doesn't work right now, need to add a setup.yml that creates EKS cluster and sets up KFP (working on that on a separate diff)
class CloudFormation:
    def __init__(self, session: boto3.Session) -> None:
        self._session = session
        self._cfn: botocore.client.BaseClient = session.client("cloudformation")

    def create_specs_file(
        self, specs_file: str, s3_bucket_name: str, efs_id: int
    ) -> None:
        username = getpass.getuser()
        rand = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
        hash = f"{username}-{rand}"
        stack_name = f"torchx-{hash}"
        this_dir = os.path.dirname(__file__)
        cfn_template = os.path.join(this_dir, "cfn/setup.yml")
        sample_specs = os.path.join(this_dir, "config/sample_specs.json")

        # TODO actually replace with params in setup.yml
        params = {
            # "WorkerRoleName": f"foobar-{hash}",
            # "RendezvousRoleName": f"barbaz-{hash}",
        }

        if s3_bucket_name:
            params["S3BucketName"] = s3_bucket_name
        if efs_id:
            params["EFSFileSystemId"] = efs_id

        self.create_stack(stack_name, cfn_template, **params)

        for _ in wait_for(
            f"cfn stack: {stack_name} to create", timeout=600, interval=2
        ):
            status, outputs = self.describe_stack(stack_name)
            if status == "CREATE_COMPLETE":
                break
            elif status == "CREATE_FAILED" or status.startswith("ROLLBACK_"):
                # when stack creation fails cfn starts rolling the stack back
                raise RuntimeError(
                    f"Error creating stack {stack_name}, status = {status}"
                )

        outputs["User"] = username

        log.info(f"Writing specs file to: {specs_file}")
        with open(sample_specs) as f:
            specs_template = Template(f.read())
            specs_template.stream(**outputs).dump(specs_file)

    def describe_stack(self, stack_name: str) -> Tuple[str, Dict[str, object]]:
        describe_res = self._cfn.describe_stacks(StackName=stack_name)

        stacks = describe_res["Stacks"]
        if len(stacks) > 1:
            raise RuntimeError(f"Found more than one stack with name {stack_name}")

        stack_desc = stacks[0]
        status = stack_desc["StackStatus"]

        # cfn outputs an array of maps, each element in the array is
        # a single output of the form "{OutputKey: <key>, OutputValue: <value>}"
        # simplify to a map of <key>, <value>  pairs
        outputs = {}
        if "Outputs" in stack_desc:
            for cfn_output in stack_desc["Outputs"]:
                key = cfn_output["OutputKey"]
                value = cfn_output["OutputValue"]
                outputs[key] = value
        return status, outputs

    def create_stack(
        self, stack_name: str, cfn_template: str, **params: Dict[str, object]
    ) -> int:
        log.info(f"Creating cloudformation stack with template: {cfn_template}")

        with open(cfn_template) as f:
            template_body = f.read()

        cfn_parameters = []
        for key, value in params.items():
            cfn_parameters.append({"ParameterKey": key, "ParameterValue": value})

        res = self._cfn.create_stack(
            StackName=stack_name,
            TemplateBody=template_body,
            Capabilities=["CAPABILITY_NAMED_IAM"],
            Parameters=cfn_parameters,
        )

        return res["StackId"]
