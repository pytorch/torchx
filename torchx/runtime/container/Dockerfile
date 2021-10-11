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

COPY . /app

RUN python setup.py install
