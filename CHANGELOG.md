# CHANGELOG

## torchx-0.1.0rc0

* `torchx.specs` API release candidate
  (still experimental but no major changes expected for `0.1.0`)
* `torchx.components`
  * made all components use docker images by default for consistency
  * removed binary_component in favor of directly writing app defs
  * `serve.torchserve` - added optional `--port` argument for upload server
  * `utils.copy` - added copy component for easy file transfer between `fsspec` path locations
  * `ddp`
    * `nnodes` no longer needs to be specified and is set from `num_replicas` instead.
    * Bug fixes.
    * End to end integration tests on Slurm and Kubernetes.
  * better unit testing support via `ComponentTestCase`.
* `torchx.schedulers`
  * Split `local` scheduler into `local_docker` and `local_cwd`.
    * For local execution `local_docker` provides the closest experience to remote behavior.
    * `local_cwd` allows reusing the same component definition for local development purposes but resolves entrypoint and deps relative to the current working directory.
  * Improvements/bug fixes to Slurm and Kubernetes schedulers.
* `torchx.pipelines`
  * `kfp` Added the ability to launch distributed apps via the new `resource_from_app` method which creates a Volcano Job from Kubeflow Pipelines.
* `torchx.runner` - general fixes and improvements around wait behavior
* `torchx.cli`
  * Improvements to output formatting to improve clarity.
  * `log` can now log from all roles instead of just one
  * `run` now supports boolean arguments
  * Experimental support for CLI being used from scripts. Exit codes are consistent and only script consumable data is logged on stdout for key commands such as `run`.
  * `--log_level` configuration flag
  * Default scheduler is now `local_docker` and decided by the first scheduler in entrypoints.
  * More robust component finding and better behavior on malformed components.
* `torchx.examples`
   * Distributed CIFAR Training Example
   * HPO
   * Improvements to lightning_classy_vision example -- uses components, datapreproc separated from injection
   * Updated to use same file directory layout as github repo
   * Added documentation on setting up kubernetes cluster for use with TorchX
   * Added distributed KFP pipeline example
* `torchx.runtime`
  * Added experimental `hpo` support with Ax (https://github.com/facebook/Ax)
  * Added experimental `tracking.ResultTracker` for distributed tracking of metrics for use with HPO.
  * Bumped pytorch version to 1.9.0.
  * Deleted deprecated storage/plugins interface.
* Docs
  * Added app/component best practices
  * Added more information on different component archetypes such as training
  * Refactored structure to more accurately represent components, runtime and
    scheduler libraries.
  * README: added information on how to install from source, nightly and different dependencies
  * Added scheduler feature compatibility matrices
  * General cleanups and improvements
* CI
  * component integration test framework
  * codecoverage
  * renamed primary branch to main
  * automated doc push
  * distributed kubernetes integration tests
  * nightly builds at https://pypi.org/project/torchx-nightly/
  * pyre now uses nightly builds
  * added slurm integration tests


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
