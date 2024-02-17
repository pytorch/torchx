FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# upgrade pip to 22.x+ which has a faster dependency resolver
RUN pip install --upgrade pip wheel --no-cache-dir

# copy requirements early so we don't have to redownload dependencies on code
# changes
COPY dev-requirements.txt /app
RUN pip install -r dev-requirements.txt --no-cache-dir

COPY . /app

RUN pip install . --no-cache-dir
