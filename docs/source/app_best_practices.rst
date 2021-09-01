App Best Practices
====================

TorchX apps can be written using any language as well as with any set of
libraries to allow for maximum flexibility. However, we do have a standard set
of recommended libraries and practices to have a starting point for users and to
provide consistency across the built in components and applications.

See :ref:`component_best_practices:Component Best Practices` for information how to handle component
management and AppDefs.

Data Passing and Storage
--------------------------

We recommend
`fsspec <https://filesystem-spec.readthedocs.io/en/latest/index.html>`__. fsspec
allows pluggable filesystems so apps can be written once and run on most
infrastructures just by changing the input and output paths.

TorchX builtin components use fsspec for all storage access to make it possible
to run in new environments by using a different fsspec backend or by adding a
new one.

Pytorch Lightning supports fsspec out of the box so using fsspec elsewhere makes
it seamless to integrate in with your trainer.

Using remote storage also makes it easier to transition your apps to running
with distributed support via libraries such as
`torch.distributed.elastic <https://pytorch.org/docs/stable/distributed.elastic.html>`__.

Train Loops
-------------

There are lots of ways to structure a training loop and it depends a lot on your
model type and architecture which is why we don't provide one out of the box.

Some common choices are:

* Pure Pytorch
* Use a managed train loop
    * `Pytorch Lighting <https://pytorch-lightning.readthedocs.io/en/latest/>`__
    * `Pytorch Ignite <https://github.com/pytorch/ignite>`__

See :ref:`components/train:Train` for more information.


Metrics
----------------

For logging metrics and monitoring your job we recommend using standalone
Tensorboard since it's supported natively by
`Pytorch tensorboard integrations <https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html>`__
and
`Pytorch Lightning logging <https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html>`__.

Since Tensorboard can log to remote storage like s3 or gcs you can view complex
information about your model while it's training.

See :ref:`components/metrics:Metrics` for more information on metric handling
within TorchX.

Checkpointing
----------------

Periodic checkpoints allow your application to recover from failures and in some
cases allow you to restart your trainer with different parameters without losing
training progress.

`Pytorch Lighting <https://pytorch-lightning.readthedocs.io/en/latest/common/weights_loading.html#checkpoint-saving>`__
provides a standardized way to checkpoint your models to an fsspec remote path.

Fine Tuning
-------------

To support things like transfer learning, fine tuning and resuming from
checkpoints we recommend having a command line argument to your app that will
resume from a checkpoint file.

This will allow you to recover from transient errors, continue train on new
data, or later adjust the learning rate without losing training progress.

Having load support allows for less code and better maintainability since you
can have one app doing a number of similar tasks.

Interpretability
----------------

We recommend `captum <https://captum.ai/>`__ for model interpretability and
analysing model results. This can be used interactively from a Jupyter notebook
or from a component.

See :ref:`components/interpret:Interpret` for more information.

Hyper Parameter Optimization
------------------------------

See :ref:`components/hpo:Hyperparameter Optimization` for more information.


Model Packaging
-----------------

The pytorch community hasn't standardized on one package format. Here's a couple
of options and when you might need to use them.

Python + Saved Weights
^^^^^^^^^^^^^^^^^^^^^^^^^

This is the most common format for packaging models. To use you'll load your
model definition from a python file and then you'll load the weights and state
dict from a `.ckpt` or `.pt` file.

This is how Pytorch Lightning's
`ModelCheckpoint <https://pytorch-lightning.readthedocs.io/en/latest/extensions/generated/pytorch_lightning.callbacks.ModelCheckpoint.html>`__ hook works.

This is generally the most common but makes it harder to make a reusable app
since your trainer app needs to include the model definition code.

TorchScript Models
^^^^^^^^^^^^^^^^^^^^^^

TorchScript is a way to create serializable and optimized Pytorch models that
can be executed without Python. This can be used for inference or training in a
performant way without relying on Python's GIL.

These model files are completely self described but not all pytorch models can
be automatically converted to TorchScript.

See the `TorchScript documentation <https://pytorch.org/docs/stable/jit.html>`__.

TorchServe Model Archiver (`.mar`)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you want to use TorchServe for inference you'll need to export your model to
this format. For inference you'll generally use a quantized version of the model
so it's best to have your trainer export both a full precision model for fine
tuning as well as a quantized `.mar` file for TorchServe to consume.

See the
`Model Archiver documentation <https://github.com/pytorch/serve/blob/master/model-archiver/README.md>`_.

torch.package
^^^^^^^^^^^^^^^^^^

This is a new format as of pytorch 1.9.0 and can be used to save and load model
definitions and their weights so you don't need to manage the model definition
separately.

See the `torch.package documentation <https://pytorch.org/docs/stable/package.html>`__.

It's quite new and doesn't have widespread adoption or support.


Serving / Inference
---------------------

For serving and inference we recommend using
`TorchServe <https://github.com/pytorch/serve>`_
for common use cases.
We provide a component that allows you to upload your model to TorchServe via
the management API.

See the :ref:`components/serve:Serve` built in components for more information.

For more complex serving and performance reasons you may need to write your own
custom inference logic. Torchscript and torch::deploy are some standard
utilities you can use to build your own inference server.

Testing
---------

Since TorchX apps are typically standard python you can write unit tests for
them like you would with any other Python code.

.. code-block:: python

    import unittest
    from your.custom.app import main

    class CustomAppTest(unittest.TestCase):
        def test_main(self) -> None:
            main(["--src", "src", "--dst", "dst"])
            self.assertTrue(...)
