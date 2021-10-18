Quickstart
==============

Running a Builtin Component
-----------------------------
Easiest way to get started with TorchX is through the provided CLI.

.. code-block:: shell-session

 # install torchx with all dependencies
 $ pip install torchx[dev]
 $ torchx --help

With TorchX you can BYO app but TorchX has a collection of builtins.
For now lets take a look at the builtins

.. code-block:: shell-session

 $ torchx builtins
 Found <n> builtin configs:
   ...
   i. utils.echo
   j. utils.touch
   ...

Echo looks familiar and simple. Lets understand how to run ``utils.echo``.

.. code-block:: shell-session

 $ torchx run --scheduler local_cwd utils.echo --help
 usage: torchx run echo [-h] [--msg MSG]

 Echos a message

 optional arguments:
 -h, --help  show this help message and exit
 --msg MSG   Message to echo

We can see that it takes a ``--msg`` argument. Lets try running it locally

.. code-block:: shell-session

 $ torchx run --scheduler local_cwd utils.echo --msg "hello world"

.. note:: ``echo`` in this context is just an app spec. It is not the application
          logic itself but rather just the "job definition" for running `/bin/echo`.
          If you haven't done so already, this is a good time to read through the
          :ref:`Basic Concepts<basics:Basics>` familiarize yourself with the basic concepts.

Defining Your Own Component
----------------------------
Now lets try to implement ``echo`` ourselves. To make things more interesting
we'll add two more parameters to our version of ``echo``:

1. Number of replicas to run in parallel
2. Prefix the message with the replica id

First we create an app spec file.
This is just a regular python file where we define the app spec.

.. code-block:: shell-session

 $ touch ~/test.py

Now copy paste the following into ``test.py``

::

 import torchx.specs as specs


 def echo(num_replicas: int, msg: str = "hello world") -> specs.AppDef:
     """
     Echos a message to stdout (calls /bin/echo)

     Args:
        num_replicas: number of copies (in parallel) to run
        msg: message to echo

     """
     return specs.AppDef(
         name="echo",
         roles=[
             specs.Role(
                 name="echo",
                 entrypoint="/bin/echo",
                 image="ubuntu:latest",
                 args=[f"replica #{specs.macros.replica_id}: {msg}"],
                 num_replicas=num_replicas,
             )
         ],
     )

Notice that

1. Unlike ``--msg``, ``--num_replicas`` does not have a default value
   indicating that it is a required argument.
2. ``test.py`` does **not** contain the logic of the app and is
   simply a job definition.


Now lets try running our custom ``echo``

.. code-block:: shell-session

 $ torchx run --scheduler local_cwd ~/test.py:echo --num_replicas 4 --msg "foobar"

 replica #0: foobar
 replica #1: foobar
 replica #2: foobar
 replica #3: foobar

Running on Other Images
-----------------------------
So far we've run ``utils.echo`` with the ``local_cwd`` scheduler. This means that the
``entrypoint`` we specified is relative to the current working directory and
ignores the specified image. That did not matter for us
since we specified an absolute path as the entrypoint (``entrypoint=/bin/echo``).
Had we specified ``entrypoint=echo`` the local_cwd scheduler would have tried to invoke
``echo`` relative to the current directory and the specified PATH.

If you have a pre-built application binary, using local_cwd is a
quick way to validate the application and the ``specs.AppDef``. But its not all
that useful if you want to run the application on a remote scheduler
(see :ref:`quickstart:Running On Other Schedulers`).

.. note:: The ``image`` string in ``specs.Role`` is an identifier to a container image
          supported by the scheduler. Refer to the scheduler documentation to find out
          what container image is supported by the scheduler you want to use.

To match remote image behavior we can use the ``local_docker`` scheduler which
will launch the image via docker and run the same application.

.. note:: Before proceeding, you will need docker installed. If you have not done so already
          follow the install instructions on: https://docs.docker.com/get-docker/

Now lets try running ``echo`` from a docker container. Modify echo's ``AppDef``
in ``~/test.py`` you created in the previous section to make the ``image="ubuntu:latest"``.

::

 import torchx.specs as specs


 def echo(num_replicas: int, msg: str = "hello world") -> specs.AppDef:
     """
     Echos a message to stdout (calls /bin/echo)

     Args:
        num_replicas: number of copies (in parallel) to run
        msg: message to echo

     """
     return specs.AppDef(
         name="echo",
         roles=[
             specs.Role(
                 name="echo",
                 entrypoint="/bin/echo",
                 image="ubuntu:latest", # IMAGE NOW POINTS TO THE UBUNTU DOCKER IMAGE
                 args=[f"replica #{specs.macros.replica_id}: {msg}"],
                 num_replicas=num_replicas,
             )
         ],
     )

Try running the echo app

.. code-block:: shell-session

 $ torchx run --scheduler local_docker \
              ~/test.py:echo \
              --num_replicas 4 \
              --msg "foobar from docker!"

Running On Other Schedulers
-----------------------------
So far we've launched components locally. Lets take a look at how to run this on
real schedulers.

.. note:: This section assumes you have already setup a running instance of
          one of the supported schedulers

Lets take a look at which schedulers we can launch into and pick one that
you have already setup.

.. code-block:: shell-session

 $ torchx schedulers

For most schedulers you will have to specify an additional ``--scheduler_args``
parameter. These are launch-time parameters to the scheduler that are associated
to the run **instance** of your application (job) rather than the job definition
(app spec) of your application, for example job ``priority``. Scheduler args
are scheduler specific so you'll have to find out what scheduler args are
required by the scheduler you are planning to use

.. code-block:: shell-session

 $ torchx runopts <sched_name>
 $ torchx runopts local_docker

Now that you've figured out what scheduler args are required, launch your app

.. code-block:: shell-session

 $ torchx run --scheduler <sched_name> --scheduler_args <k1=v1,k2=v2,...> \
     utils.sh ~/my_app.py <app_args...>
 $ torchx run --scheduler local_cwd --scheduler_args log_dir=/tmp \
     utils.sh ~/my_app.py --foo=bar

.. note:: If your app args overlap with the ``run`` subcommand's args, you
          have to use the ``--`` delimiter for argparse to not get confused.
          For example, if your app also takes a ``--scheduler`` argument,
          run it as shown below.

.. code-block:: shell-session

 $ torchx run --scheduler local_docker ~/my_app.py -- --scheduler foobar


Next Steps
------------
1. Checkout other features of the :ref:`torchx CLI<cli:CLI>`
2. Learn how to author more complex app specs by referencing :ref:`specs:torchx.specs`
3. Browse through the collection of :ref:`builtin components<index:Components Library>`
4. Take a look at the :ref:`list of schedulers<Schedulers>` supported by the runner
5. See which :ref:`ML pipeline platforms<Pipelines>` you can run components on
