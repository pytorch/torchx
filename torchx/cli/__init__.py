# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
The ``torchx`` CLI is a commandline tool around :py:class:`torchx.runner.Runner`.
It allows users to launch :py:class:`torchx.specs.AppDef` directly onto
one of the supported schedulers without authoring a pipeline (aka workflow).
This is convenient for quickly iterating on the application logic without
incurring both the technical and cognitive overhead of learning, writing, and
dealing with pipelines.


.. note:: When in doubt use ``torchx --help``.

Listing the builtin components
----------------------------------
Most of the components under the ``torchx.components`` module are what the CLI
considers "built-in" apps. Before you write your own component you should
browse through the builtins to see if there is one that fits your needs already.
If so, no need to even author an app spec!

.. code-block:: shell-session

 $ torchx builtins
 Found <n> builtin configs:
  1. echo
  2. simple_example
  3. touch
  ... <omitted for brevity>

Listing the supported schedulers
----------------------------------
To get a list of supported schedulers that you can launch your job into run:

.. code-block:: shell-session

 $ torchx schedulers

Running an app as a job
---------------------------
The ``run``  subcommand takes either one of:

1. name of the builtin

   .. code-block:: shell-session

    $ torchx run --scheduler <sched_name> echo

2. full python module path of the component function

   .. code-block:: shell-session

     $ torchx run --scheduler <sched_name> torchx.components.utils.echo

3. file path of the ``*.py`` file the defines the component
   along with the component function name in that file.

   .. code-block:: shell-session

     $ cat ~/my_trainer_spec.py
     import torchx.specs as specs

     def get_spec(foo: int, bar: str) -> specs.AppDef:
       <...spec file details omitted for brevity...>

     $ torchx run --scheduler <sched_name> ~/my_trainer_spec.py get_spec

Now that you have understood how to chose which app to launch, now its time
to see what the parameters need to be passed. There are three sets of
parameters:

1. arguments to the ``run`` subcommand, see a list of them by running:

   .. code-block:: shell-session

    $ torchx run --help

2. arguments to the scheduler (``--scheduler_args``, also known as ``run_options`` or ``run_configs``),
   each scheduler takes different args, to find out the args for a specific scheduler run (command for
   ``local`` scheduler shown below:

   .. code-block:: shell-session

    $ torchx runopts local
    { 'image_fetcher': { 'default': 'dir',
                     'help': 'image fetcher type',
                     'type': 'str'},
    'log_dir': { 'default': 'None',
               'help': 'dir to write stdout/stderr log files of replicas',
               'type': 'str'}}

    $ torchx run --scheduler local --scheduler_args image_fetcher=dir,log_dir=/tmp ...

3. arguments to the component (the app args are included here), this also depends on the
   component and can be seen with the ``--help`` string on the component

   .. code-block:: shell-session

    $ torchx run --scheduler local echo --help
    usage: torchx run echo.torchx [-h] [--msg MSG]

    Echos a message

    optional arguments:
    -h, --help  show this help message and exit
    --msg MSG   Message to echo

Putting everything together, running ``echo`` with the ``local`` scheduler:

.. code-block:: shell-session

 $ torchx run \
     --scheduler local \
     --scheduler_args image_fetcher=dir,log_dir=/tmp \
     echo \
     --msg "hello $USER"
 === RUN RESULT ===
 Launched app: local://torchx_kiuk/echo_ecd30f74

The ``run`` subcommand does not block for the job to finish, instead it simply
schedules the job on the specified scheduler and prints an ``app handle``
which is a URL of the form: ``$scheduler_name://torchx_$user/$job_id``.
Keep note of this handle since this is what you'll need to provide to other
subcommands to identify your job.

Describing and querying the status of a job
---------------------------------------------
The ``describe`` subcommand is essentially the inverse of the ``run`` command.
That is, it prints the app spec given an ``app_handle``.

.. code-block:: shell-session

 $ torchx describe <app handle>


.. important:: The ``describe`` command attempts to recreate an app spec
               by querying the scheduler for the job description. So what you
               see printed is not always 100% the exact same app spec that was
               given to the ``run``. The extent to which the runner can
               recreate the app spec depends on numerous factors such as
               how descriptive the scheduler's ``describe_job`` API is as well
               as whether there were fields in the app spec that were ignored
               when submitting the job to the scheduler because the scheduler
               does not have support for such a parameter/functionality.
               NEVER rely on the ``describe`` API as a storage function for
               your app spec. It is simply there to help you spot check things.

To get the status of a running job:

.. code-block:: shell-session

 $ torchx status <app_handle> # prints status for all replicas and roles
 $ torchx status --role trainer <app_handle> # filters it down to the trainer role

Filtering by ``--role`` is useful for large jobs that have multiple roles.

Viewing Logs
---------------------

.. note:: This functionality depends on how long your scheduler setup retains logs
          TorchX DOES NOT archive logs on your behalf, but rather relies on the scheduler's
          ``get_log`` API to obtain the logs. Refer to your scheduler's user manual
          to setup log retention properly.

The ``log`` subcommand is a simple wrapper around the scheduler's ``get_log`` API
to let you pull/print the logs for all replicas and roles from one place.
It also lets you pull replica or role specific logs. Below are a few log access
patterns that are useful and self explanatory

.. code-block:: shell-session

 $ torchx log <app_handle>
 Prints all logs from all replicas and roles (each log line is prefixed with role name and replica id)

 $ torchx log --tail <app_handle>
 If the job is still running tail the logs

 $ torchx log --regex ".*Exception.*" <app_handle>
 regex filter to exceptions

 $ torchx log <app_handle>/<role>
 $ torchx log <app_handle>/trainer
 pulls all logs for the role trainer

 $ torchx log <app_handle>/<role_name>/<replica_id>
 $ torchx log <app_handle>/trainer/0,1
 only pulls trainer 0 and trainer 1 (node not rank) logs


.. note:: Some schedulers do not support server-side regex filters. In this case
          the regex filter is applied on the client-side, meaning the full logs
          will have to be passed through the client. This may be very taxing to
          the local host. Please use your best judgement when using the logs API.

"""
