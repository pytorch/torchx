FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

RUN pip install fsspec[s3]

WORKDIR /app

# copy requirements early so we don't have to redownload dependencies on code
# changes
COPY dev-requirements.txt /app
RUN pip install --upgrade -r dev-requirements.txt

COPY . /app

RUN python setup.py install
