#!/bin/bash

set -eux
minikube delete
minikube start --driver=docker --cpus=max --memory=max --nodes=2
minikube addons enable registry

# setup multi node volumes
# https://github.com/kubernetes/minikube/issues/12360#issuecomment-1430243861
minikube addons disable storage-provisioner
minikube addons disable default-storageclass
minikube addons enable volumesnapshots
minikube addons enable csi-hostpath-driver
kubectl patch storageclass csi-hostpath-sc -p '{"metadata": {"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'

# create namespace
kubectl create namespace torchx-dev

# install Kueue and Kueue related resources
VERSION=v0.6.0
kubectl apply --server-side -f https://github.com/kubernetes-sigs/kueue/releases/download/$VERSION/manifests.yaml

# Function to check if the kueue manager pod is running
check_pod_status() {
    local status=$(kubectl get pods -n kueue-system | grep "kueue-controller-manager" | awk '{print $3}')
    echo "$status"
}

# Wait until the pod is in the 'Running' state
echo "Waiting for kueue-controller-manager pod to be running in the kueue-system namespace..."
while [[ $(check_pod_status) != "Running" ]]; do
    sleep 5
done
# Function to check if the service exists
check_service_existence() {
  kubectl get svc kueue-webhook-service -n kueue-system --no-headers 2>/dev/null
}

# Wait until the service exists
echo "Waiting for kueue-webhook-service to exist in the kueue-system namespace..."
while [[ $(check_service_existence) == "" ]]; do
  sleep 5
done
echo "kueue-webhook-service exists in the kueue-system namespace."
sleep 20
# Create Cluster Queue - UPDATE MAX VALUES
cat <<EOF | kubectl apply --server-side -f - 
    apiVersion: kueue.x-k8s.io/v1beta1
    kind: ClusterQueue
    metadata:
        name: "cluster-queue"
    spec:
      namespaceSelector: {} # match all.
      resourceGroups:
      - coveredResources: ["cpu", "memory", "pods"]
        flavors:
        - name: "default-flavor"
          resources:
          - name: "cpu"
            nominalQuota: 16
          - name: "memory"
            nominalQuota: 64000Mi
          - name: "pods"
            nominalQuota: 5
EOF
echo "Cluster Queue: cluster-queue applied!"

echo "Applying Resource Flavor"
cat <<EOF | kubectl apply --server-side -f - 
    apiVersion: kueue.x-k8s.io/v1beta1
    kind: ResourceFlavor
    metadata:
        name: default-flavor
EOF
echo "Resource Flavour: default-flavor applied!"

cat <<EOF | kubectl apply --server-side -f - 
    apiVersion: kueue.x-k8s.io/v1beta1
    kind: LocalQueue
    metadata:
        namespace: torchx-dev
        name: torchx-local-queue
    spec:
      clusterQueue: cluster-queue
EOF
echo "Local Queue: torchx-local-queue applied!" 

# portforwarding
kubectl port-forward --namespace kube-system service/registry 5000:80 &

