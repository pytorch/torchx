Quickstart
==============

Running a Builtin Component
-----------------------------
Easiest way to get started with TorchX is through the provided CLI.

.. code-block:: shell-session

 $ pip install torchx
 $ torchx --help

With TorchX you can BYO app but TorchX has a collection of builtins.
For now lets take a look at the builtins

.. code-block:: shell-session

 $ torchx builtins
 Found <n> builtin configs:
   1. echo
   2. touch
   ...

Echo looks familiar and simple. Lets understand how to run ``echo``.

.. code-block:: shell-session

 $ torchx run --scheduler local echo --help
 usage: torchx run echo [-h] [--msg MSG]

 Echos a message

 optional arguments:
 -h, --help  show this help message and exit
 --msg MSG   Message to echo

We can see that it takes a ``--msg`` argument. Lets try running it locally

.. code-block:: shell-session

 $ torchx run --scheduler local echo --msg "hello world"

.. note:: ``echo`` in this context is just an app spec. It is not the application
          logic itself but rather just the "job definition" for running `/bin/echo`.
          If you haven't done so already, this is a good time to read through the
          :ref:`Basic Concepts<Basics>` familiarize yourself with the basic concepts.

Defining Your Own Component
----------------------------
Now lets try to implement ``echo`` ourselves. To make things more interesting
we'll add two more parameters to our version of ``echo``:

1. Number of replicas to run in parallel
2. Prefix the message with the replica id

First we create an app spec file.
This is just a regular python file where we define the app spec.

.. code-block:: shell-session

 $ touch ~/echo_torchx.py

Now copy paste the following into echo_torchx.py

::

 import torchx.specs as specs


 def get_app_spec(num_replicas: int, msg: str = "hello world") -> specs.AppDef:
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
                 args=[f"replica #{specs.macros.replica_id}: msg"],
                 container=specs.Container(image="/tmp"),
                 num_replicas=1,
             )
         ],
     )

Notice that

1. Unlike ``--msg``, ``--num_replicas`` does not have a default value
   indicating that it is a required argument.
2. We use a local dir (``/tmp``) as the ``image``. In practice this will be
   the identifier of the package (e.g. Docker image) that the scheduler supports.
3. ``echo_torchx.py`` does **not** contain the logic of the app and is
   simply a job definition.


Now lets try running our custom ``echo``

.. code-block:: shell-session

 $ torchx run --scheduler local ~/echo_torchx.py --num_replicas 4 --msg "foobar"

 replica #0: foobar
 replica #1: foobar
 replica #2: foobar
 replica #3: foobar

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
 $ torchx runopts local

Now that you've figured out what scheduler args are required, launch your app

.. code-block:: shell-session

 $ torchx run --scheduler <sched_name> --scheduler_args <k1=v1,k2=v2,...> \
     ~/my_app.py <app_args...>
 $ torchx run --scheduler local --scheduler_args image_type=dir,log_dir=/tmp \
     ~/my_app.py --foo=bar

.. note:: If your app args overlap with the ``run`` subcommand's args, you
          have to use the ``--`` delimiter for argparse to not get confused.
          For example, if your app also takes a ``--scheduler`` argument,
          run it as shown below.

.. code-block:: shell-session

 $ torchx run --scheduler local ~/my_app.py -- --scheduler foobar


Next Steps
------------
1. Checkout other features of the :ref:`torchx CLI<CLI>`
2. Learn how to author more complex app specs by referencing :ref:`torchx.specs`
3. Browse through the collection of :ref:`builtin components<Components Library>`
4. Take a look at the :ref:`list of schedulers<Schedulers>` supported by the runner
5. See which :ref:`ML pipeline platforms<Pipelines>` you can run components on
