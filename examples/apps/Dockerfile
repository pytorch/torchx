FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

# Note: the pytorch image installs version of the torchvision that is not compatible
# with CUDA. By using it directly there will be a segfault.
RUN pip uninstall -y torchvision
RUN pip install torchvision

RUN pip install classy_vision pytorch-lightning fsspec[s3] torch-model-archiver captum boto3

WORKDIR /app

COPY . /app
