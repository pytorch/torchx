# CHANGELOG

## torchx-0.1.0b0

* `torchx.specs` API release candidate
  (still experimental but no major changes expected for `0.1.0`)

* `torchx.pipelines` - Kubeflow Pipeline adapter support
* `torchx.runner` - SLURM and local scheduler support
* `torchx.components` - several utils, ddp, torchserve builtin components
* `torchx.examples`
   * Colab support for examples
   * `apps`:
     * classy vision + lightning trainer
     * torchserve deploy
     * captum model visualization
   * `pipelines`:
     * apps above as a Kubeflow Pipeline
     * basic vs advanced Kubeflow Pipeline examples
* CI
  * unittest, pyre, linter, KFP launch, doc build/test
