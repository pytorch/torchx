FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

RUN pip install fsspec[s3]

# Note: the pytorch image installs version of the torchvision that is not compatible
# with CUDA. By using it directly there will be a segfault.
RUN pip uninstall -y torchvision
RUN pip install torchvision

WORKDIR /app

# copy requirements early so we don't have to redownload dependencies on code
# changes
COPY dev-requirements.txt /app
RUN pip install -r dev-requirements.txt

# TODO: Remove this and get dependency on torch 1.10 when it is released

RUN pip uninstall -y torch

RUN pip install https://download.pytorch.org/whl/test/cu113/torch-1.10.0%2Bcu113-cp37-cp37m-linux_x86_64.whl

RUN pip uninstall -y torchtext

RUN pip install https://download.pytorch.org/whl/test/torchtext-0.10.0-cp37-cp37m-linux_x86_64.whl

COPY . /app

RUN python setup.py install
