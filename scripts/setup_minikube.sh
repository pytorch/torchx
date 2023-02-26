#!/bin/bash

set -eux
minikube delete
minikube start --driver=docker --cpus=max --memory=max --nodes=2
minikube addons enable registry
kubectl apply -f https://raw.githubusercontent.com/volcano-sh/volcano/v1.7.0/installer/volcano-development.yaml
kubectl create namespace torchx-dev
kubectl port-forward --namespace kube-system service/registry 5000:80 &
