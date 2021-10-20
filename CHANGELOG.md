# CHANGELOG

## torchx-0.1.0

* `torchx.schedulers`
  * `local_docker` print a nicer error if Docker is not installed #284
* `torchx.cli`
  *  Improved error messages when `-cfg` is not provided #271
* `torchx.components`
  * Update `dist.ddp` to use `c10d` backend as default #263
* `torchx.aws`
  * Removed entirely as it was unused

* Docs
  * Restructure documentation to be more clear
  * Merged Hello World example with the Quickstart guide to reduce confusion
  * Updated Train / Distributed component documentation
  * Renamed configure page to "Advanced Usage" to avoid confusion with experimental .torchxconfig
  * Renamed Localhost page to just Local to better match the class name
  * Misc cleanups / improvements

* Tests
  * Fixed test failure when no secrets are present #274
  * Added macOS variant to our unit tests #209

## torchx-0.1.0rc1

* `torchx.specs`
  * base_image has been deprecated
  * Some predefined AWS specific named_resources have been added
  * Docstrings are no longer required for component definitions to make it
  easier to write them. They will be still rendered as help text if present and
  are encouraged but aren't required.
  * Improved vararg handling logic for components

* `torchx.runner`
  * Username has been removed from the session name
  * Standardized `runopts` naming

* `torchx.cli`
  * Added experimental `.torchxconfig` file which can be used to set default
  scheduler arguments for all runs.
  * Added `--version` flag
  * `builtins` ignores `torchx.components.base` folder

* Docs
  * Improved entry_points and resources docs
  * Better component documentation
  * General improvements and fixes

* Examples
  * Moved examples to be under torchx/ and merged the examples container with
  the primary container to simplify usage.
  * Added a self contained "Hello World" example
  * Switched lightning_classy_vision example to use ResNet model architecture so
  it will actually converage
  * Removed CIFAR example and merged functionality into lightning_classy_vision

* CI
  * Switched to OpenID Connect based auth

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
