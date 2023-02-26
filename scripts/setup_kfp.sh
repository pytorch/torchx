#!/bin/bash

set -eux

export PIPELINE_VERSION=1.8.5
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/cluster-scoped-resources?ref=$PIPELINE_VERSION"
kubectl wait --for condition=established --timeout=60s crd/applications.app.k8s.io
kubectl apply -k "github.com/kubeflow/pipelines/manifests/kustomize/env/dev?ref=$PIPELINE_VERSION"
kubectl apply -f resources/kfp_volcano_role_binding.yaml
kubectl wait --namespace=kubeflow --for condition=available --timeout=10m deployments/metadata-grpc-deployment
