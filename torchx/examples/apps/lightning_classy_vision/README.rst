Lightning + Classy Vision Trainer Example
#########################################

This example consists of model training and interpretability apps that uses
PyTorch Lightning and ClassyVision. The apps have shared logic so are split
across several files.

The trainer and interpret apps do not have any TorchX-isms and are
simply ClassyVision (PyTorch) and Captum applications. TorchX helps you
run these applications on various schedulers and localhost.
The trainer app is a distributed data parallel style application and is launched
with the `dist.ddp` built-in. The interpret app is a single node application
and is launched as a regular python process with the `utils.python` built-in.

For instructions on how to run these apps with TorchX refer to the documentations
in their respective main modules: `train.py` and `interpret.py`.

