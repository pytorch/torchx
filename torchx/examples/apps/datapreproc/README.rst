Data Preprocessing Example
##########################

This is a simple TorchX app that downloads some data via HTTP, normalizes the
images via torchvision and then reuploads it via fsspec.

This examples has two Python files: the app which actually does the
preprocessing and the component definition which can be used with TorchX to
launch the app.
