# Copyright (c) Meta Platforms, Inc. and affiliates.
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
from typing import List, Optional, Tuple

import fsspec
import pytorch_lightning as pl
import torch
import torch.jit
from torch.nn import functional as F
from torchmetrics import Accuracy
from torchvision.models.resnet import BasicBlock, ResNet


class TinyImageNetModel(pl.LightningModule):
    """
    An very simple linear model for the tiny image net dataset.
    """

    def __init__(
        self, layer_sizes: Optional[List[int]] = None, lr: Optional[float] = None
    ) -> None:
        super().__init__()

        if not layer_sizes:
            layer_sizes = [1, 1, 1, 1]

        self.lr: float = lr or 0.001

        # We use the torchvision resnet model with some small tweaks to match
        # TinyImageNet.
        m = ResNet(BasicBlock, layer_sizes)
        m.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        m.fc.out_features = 200
        self.model: ResNet = m

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    # pyre-fixme[14]
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

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
        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)
        self.log(f"{step_name}_loss", loss)
        acc_metric(y_pred, y)
        self.log(f"{step_name}_acc", acc_metric.compute())
        return loss

    # pyre-fixme[3]: TODO(aivanou): Figure out why oss pyre can identify type but fb cannot.
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


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
            "torchx/examples/apps/lightning/handler/handler.py",
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


# sphinx_gallery_thumbnail_path = '_static/img/gallery-lib.png'
