#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

#
# Builds docs from the checkedout HEAD
# and pushes the artifacts to gh-pages branch in github.com/pytorch/torchx
#
# 1. sphinx generated docs are copied to <repo-root>/<version>
# 2. if a release tag is found on HEAD then redirects are copied to <repo-root>/latest
# 3. if no release tag is found on HEAD then redirects are copied to <repo-root>/master
#
# gh-pages branch should look as follows:
# <repo-root>
#           |- 0.1.0rc2
#           |- 0.1.0rc3
#           |- <versions...>
#           |- master (redirects to the most recent ver in trunk, including release)
#           |- latest (redirects to the most recent release)
# If the most recent  release is 0.1.0 and master is at 0.1.1rc1 then,
# https://pytorch.org/torchx/master -> https://pytorch.org/torchx/0.1.1rc1
# https://pytorch.org/torchx/latest -> https://pytorch.org/torchx/0.1.0
#
# Redirects are done via Jekyll redirect-from  plugin. See:
#   sources/scripts/create_redirect_md.py
#   Makefile (redirect target)
#  (on gh-pages branch) _layouts/docs_redirect.html

dry_run=0
for arg in "$@"; do
    shift
    case "$arg" in
        "--dry-run") dry_run=1 ;;
        "--help") echo "Usage $0 [--dry-run]"; exit 0 ;;
    esac
done

repo_root=$(git rev-parse --show-toplevel)
branch=$(git rev-parse --abbrev-ref HEAD)
commit_id=$(git rev-parse --short HEAD)

if ! release_tag=$(git describe --tags --exact-match HEAD 2>/dev/null); then
    echo "No release tag found, building docs for master..."
    redirects=(master)
    release_tag="master"
else
    echo "Release tag $release_tag found, building docs for release..."
    redirects=(latest master)
fi

echo "Installing torchx from $repo_root/torchx..."
cd "$repo_root/torchx" || exit
pip uninstall -y torchx
python setup.py install

torchx_ver=$(python -c "import torchx; print(torchx.__version__)")

echo "Building torchx-$torchx_ver docs..."
docs_dir=$repo_root/docs
build_dir=$docs_dir/build
cd "$docs_dir" || exit
pip install -r requirements.txt
make clean html
echo "Doc build complete"

if [ $dry_run -eq 1 ]; then
    echo "*** dry-run mode, building only. See build artifacts in: $build_dir"
    exit
fi

tmp_dir=/tmp/torchx_docs_tmp
rm -rf "${tmp_dir:?}"

echo "Checking out gh-pages branch..."
gh_pages_dir="$tmp_dir/torchx_gh_pages"
git clone -b gh-pages --single-branch git@github.com:pytorch/torchx.git  $gh_pages_dir

echo "Copying doc pages for $torchx_ver into $gh_pages_dir..."
rm -rf "${gh_pages_dir:?}/${torchx_ver:?}"
cp -R "$build_dir/$torchx_ver/html" "$gh_pages_dir/$torchx_ver"

for redirect in "${redirects[@]}"; do
  echo "Copying redirects for $redirect -> $torchx_ver..."
  rm -rf "${gh_pages_dir:?}/${redirect:?}"
  cp -R "$build_dir/redirects" "$gh_pages_dir/$redirect"
done

if [ "$release_tag" != "master" ]; then
    echo "Copying redirects for default(latest) -> $torchx_ver..."
    cp -R "$build_dir/redirects/." "$gh_pages_dir"
fi

cd $gh_pages_dir || exit
git add .
git commit --quiet -m "[doc_push][$release_tag] built from $commit_id ($branch). Redirects: ${redirects[*]} -> $torchx_ver."
