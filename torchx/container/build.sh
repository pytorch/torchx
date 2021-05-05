#!/bin/bash

set -ex

tar -czh --exclude docs . | docker build -t torchx - -f torchx/container/Dockerfile
