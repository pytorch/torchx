FROM pytorch/pytorch

RUN pip install fsspec[s3]

WORKDIR /app

# copy requirements early so we don't have to redownload dependencies on code
# changes
COPY requirements.txt /app
RUN pip install -r requirements.txt

COPY . /app

RUN python setup.py install
