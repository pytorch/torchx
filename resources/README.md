The readme describes how to create and delete eks cluster and kfp services.

#### Creating EKS cluster

    eksctl create cluster -f torchx-dev-eks.yml

#### Creating KFP

    kfctl apply -V -f torchx-dev-kfp.yml

#### Applying kfp role binding

    kubectl apply -f kfp_volcano_role_binding.yaml

#### Creating torchserve

    https://github.com/pytorch/serve/tree/master/kubernetes/EKS

#### Installing volcano

    kubectl apply -f https://raw.githubusercontent.com/volcano-sh/volcano/master/installer/volcano-development.yaml

    Install `vcctl`


#### Installing kfp from source code

    Source doc: https://www.kubeflow.org/docs/components/pipelines/installation/standalone-deployment/

    kubectl apply -k manifests/kustomize/cluster-scoped-resources

    kubectl apply -k manifests/kustomize/env/dev


#### Starting etcd service

    kubectl apply -f etcd.yaml

#### Deleting KFP services

    cd torchx-dev-1-18 && kfctl delete -V -f torchx-dev-kfp.yml

#### Deleting EKS cluster

    eksctl delete -f torch-dev-eks.yml

This command most likely will fail. EKS user cloudformation to create many resources, that
are hard to remove. If the command fails there needs to be done manual cleanup:
* Clean up the associated VPC. Go to AWS Console -> VPC -> Press `Delete`. This will
point you the ENI and NAT that needs to be deleted manually.
* Clean up the cloudformation temalte. Go to AWS Console -> CNF -> delete corresponding templates.

### Gotchas:

* The directory where `torchx-dev-kfp.yml` is located should be the same name
    as eks cluster

* The node groups in eks cluster HAVE to be spread more than a single AZ, otherwise there
 will be problems with `istio-ingress`

* KFP troubleshooting: https://www.kubeflow.org/docs/distributions/aws/troubleshooting-aws/

* Enable Kubernetes nodes to access AWS account resources: https://stackoverflow.com/a/64617080/1446208

* Torchserve fails with `DownloadArchiveException` : https://github.com/pytorch/serve/issues/1218
