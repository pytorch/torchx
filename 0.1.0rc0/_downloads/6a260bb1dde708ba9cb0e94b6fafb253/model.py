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
from typing import Tuple, Optional, List

import fsspec
import pytorch_lightning as pl
import torch
import torch.jit
from torch.nn import functional as F
from torchmetrics import Accuracy


class TinyImageNetModel(pl.LightningModule):
    """
    An very simple linear model for the tiny image net dataset.
    """

    def __init__(self, layer_sizes: Optional[List[int]] = None) -> None:
        super().__init__()

        # build a model with hidden layers specified by layer_sizes
        if layer_sizes is None:
            layer_sizes = []
        dims = [64 * 64] + layer_sizes + [4096]
        layers = []
        for i, (a, b) in enumerate(zip(dims, dims[1:])):
            if i > 0:
                layers.append(torch.nn.ReLU(inplace=True))
            layers.append(torch.nn.Linear(a, b))
        self.seq: torch.nn.modules.Sequential = torch.nn.Sequential(*layers)

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    # pyre-fixme[14]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x.view(x.size(0), -1))

    # pyre-fixme[14]
    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step("train", self.train_acc, batch, batch_idx)

    # pyre-fixme[14]
    def validation_step(
        self, val_batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self._step("val", self.val_acc, val_batch, batch_idx)

    def _step(
        self,
        step_name: str,
        acc_metric: Accuracy,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        x, y = batch
        output = self(x)
        _, y_pred = torch.max(output, dim=1)
        loss = F.cross_entropy(output, y)
        self.log(f"{step_name}_loss", loss)
        acc_metric(y_pred, y)
        self.log(f"{step_name}_acc", acc_metric.compute())
        return loss

    # pyre-fixme[3]: TODO(aivanou): Figure out why oss pyre can identify type but fb cannot.
    def configure_optimizers(self):
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
            "examples/apps/lightning_classy_vision/handler/handler.py",
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
