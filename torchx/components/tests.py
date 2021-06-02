# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import torchx.specs as specs


def echo(msg: str = "hello world", image: str = "/tmp") -> specs.Application:
    """Echos a message

    Args:
        msg: Message to echo
        image: Image to run
    """

    echo = specs.Role(
        name="echo",
        entrypoint="/bin/echo",
        args=[msg],
        container=specs.Container(image=image),
        num_replicas=1,
    )

    return specs.Application(name="echo", roles=[echo])


def touch(file: str) -> specs.Application:
    """Test component, creates a file in tmp dir

    Args:
        file: File to touch
    """

    touch = specs.Role(
        name="touch",
        entrypoint="/bin/touch",
        args=[f"{file}"],
        container=specs.Container(image="/tmp"),
        num_replicas=1,
    )

    return specs.Application(name="touch", roles=[touch])


def touch_v2(file: str) -> specs.Application:
    """Test component, creates a file in tmp dir

    Args:
        file: File to touch
    """

    touch = specs.Role(
        name="touch",
        entrypoint="/bin/touch",
        args=[f"{file}.testv2"],
        container=specs.Container(image="/tmp"),
        num_replicas=1,
    )

    return specs.Application(name="touch", roles=[touch])


def simple(
    num_trainers: int = 10, trainer_image: str = "pytorch/torchx:latest"
) -> specs.Application:
    """A simple configuration example.

    Args:
        num_trainers: The number of trainers to use.
        trainer_image: The trainer image to use.
    """

    trainer_container = specs.Container(image=trainer_image)
    reader_container = specs.Container(image=trainer_image)

    trainer = specs.Role(
        name="trainer",
        entrypoint="train_main.py",
        args=["--epochs", "50"],
        env={"MY_ENV_VAR": "foobar"},
        container=trainer_container,
        num_replicas=num_trainers,
    )

    ps = specs.Role(
        name="parameter_server",
        entrypoint="ps_main.py",
        container=trainer_container,
        num_replicas=10,
    )

    reader = specs.Role(
        name="reader",
        entrypoint="reader_main.py",
        args=["--buffer_size", "1024"],
        container=reader_container,
        num_replicas=1,
    )

    return specs.Application(name="my_train_job", roles=[trainer, ps, reader])
