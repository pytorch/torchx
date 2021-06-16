# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tiny ImageNet Model
====================

This is a toy model for doing regression on the tiny imagenet dataset. It's used
by the apps in the same folder.
"""

import os.path
import subprocess
from typing import Tuple

import fsspec
import pytorch_lightning as pl
import torch.jit
from torch.nn import functional as F


class TinyImageNetModel(pl.LightningModule):
    """
    An very simple linear model for the tiny image net dataset.
    """

    def __init__(self) -> None:
        super().__init__()
        self.l1 = torch.nn.Linear(64 * 64, 4096)

    # pyre-fixme[14]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    # pyre-fixme[14]
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_nb: int
    ) -> torch.Tensor:
        x, y = batch
        loss = F.cross_entropy(self(x), y)
        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.parameters(), lr=0.02)


def export_inference_model(
    model: TinyImageNetModel, out_path: str, tmpdir: str
) -> None:
    """
    export_inference_model uses TorchScript JIT to serialize the
    TinyImageNetModel into a standalone file that can be used during inference.
    TorchServe can also handle interpreted models with just the model.py file if
    your model can't be JITed.
    """

    print("exporting inference model")
    jit_path = os.path.join(tmpdir, "model_jit.pt")
    jitted = torch.jit.script(model)
    print(f"saving JIT model to {jit_path}")
    torch.jit.save(jitted, jit_path)

    model_name = "tiny_image_net"

    mar_path = os.path.join(tmpdir, f"{model_name}.mar")
    print(f"creating model archive at {mar_path}")
    subprocess.run(
        [
            "torch-model-archiver",
            "--model-name",
            "tiny_image_net",
            "--handler",
            "lightning_classy_vision/handler/handler.py",
            "--version",
            "1",
            "--serialized-file",
            jit_path,
            "--export-path",
            tmpdir,
        ],
        check=True,
    )

    remote_path = os.path.join(out_path, "model.mar")
    print(f"uploading to {remote_path}")
    fs, _, rpaths = fsspec.get_fs_token_paths(remote_path)
    assert len(rpaths) == 1, "must have single path"
    fs.put(mar_path, rpaths[0])
