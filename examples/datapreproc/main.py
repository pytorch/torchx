#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
import sys
import tarfile
import tempfile
from typing import List

import fsspec
from PIL import Image
from torchvision import transforms
from torchvision.datasets.folder import is_image_file
from torchvision.datasets.utils import download_and_extract_archive
from tqdm import tqdm


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="example data preprocessing",
    )
    parser.add_argument(
        "--input_path",
        type=str,
        help="dataset to download",
        default="http://cs231n.stanford.edu/tiny-imagenet-200.zip",
    )
    parser.add_argument(
        "--input_md5",
        type=str,
        help="dataset to download",
        default="90528d7ca1a48142e341f4ef8d21d0de",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help="remote path to save the .tar.gz data to",
        required=True,
    )
    return parser.parse_args(argv)


def main(argv: List[str]) -> None:
    args = parse_args(argv)
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"downloading {args.input_path} to {tmpdir}...")
        download_and_extract_archive(args.input_path, tmpdir, md5=args.input_md5)

        img_root = os.path.join(
            tmpdir,
            os.path.splitext(os.path.basename(args.input_path))[0],
        )
        print(f"img_root={img_root}")

        print("transforming images...")
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),
                transforms.ToPILImage(),
            ]
        )

        image_files = []
        for root, _, fnames in os.walk(img_root):
            for fname in fnames:
                path = os.path.join(root, fname)
                if not is_image_file(path):
                    continue
                image_files.append(path)

        for path in tqdm(image_files):
            f = Image.open(path)
            f = transform(f)
            f.save(path)

        tar_path = os.path.join(tmpdir, "out.tar.gz")
        print(f"packing images into {tar_path}...")
        with tarfile.open(tar_path, mode="w:gz") as f:
            f.add(img_root, arcname="")

        print(f"uploading dataset to {args.output_path}...")
        fs, _, rpaths = fsspec.get_fs_token_paths(args.output_path)
        assert len(rpaths) == 1, "must have single output path"
        fs.put(tar_path, rpaths[0])


if __name__ == "__main__":
    main(sys.argv[1:])
