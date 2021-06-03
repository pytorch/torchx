FROM pytorch/pytorch

RUN pip install classy_vision pytorch-lightning fsspec[s3] torch-model-archiver

WORKDIR /app

COPY . /app
