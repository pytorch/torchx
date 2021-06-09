# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torchx.specs as specs


def echo(msg: str = "hello world", image: str = "/tmp") -> specs.AppDef:
    """Echos a message

    Args:
        msg: Message to echo
        image: Image to run
    """

    echo = specs.Role(
        name="echo",
        image=image,
        entrypoint="/bin/echo",
        args=[msg],
        num_replicas=1,
    )

    return specs.AppDef(name="echo", roles=[echo])


def touch(file: str) -> specs.AppDef:
    """Echos a message

    Args:
        file: File to touch
    """

    touch = specs.Role(
        name="touch",
        image="/tmp",
        entrypoint="/bin/touch",
        args=[f"{file}.test"],
        num_replicas=1,
    )

    return specs.AppDef(name="touch", roles=[touch])


def touch_v2(file: str) -> specs.AppDef:
    """Echos a message

    Args:
        file: File to touch
    """

    touch = specs.Role(
        name="touch",
        image="/tmp",
        entrypoint="/bin/touch",
        args=[f"{file}.testv2"],
        num_replicas=1,
    )

    return specs.AppDef(name="touch", roles=[touch])


def simple(
    num_trainers: int = 10, trainer_image: str = "pytorch/torchx:latest"
) -> specs.AppDef:
    """A simple configuration example.

    Args:
        num_trainers: The number of trainers to use.
        trainer_image: The trainer image to use.
    """

    trainer = specs.Role(
        name="trainer",
        image=trainer_image,
        entrypoint="train_main.py",
        args=["--epochs", "50"],
        env={"MY_ENV_VAR": "foobar"},
        num_replicas=num_trainers,
    )

    ps = specs.Role(
        name="parameter_server",
        image=trainer_image,
        entrypoint="ps_main.py",
        num_replicas=10,
    )

    reader = specs.Role(
        name="reader",
        image=trainer_image,
        entrypoint="reader_main.py",
        args=["--buffer_size", "1024"],
        num_replicas=1,
    )

    return specs.AppDef(name="my_train_job", roles=[trainer, ps, reader])
