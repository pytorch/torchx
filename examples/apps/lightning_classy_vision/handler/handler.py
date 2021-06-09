# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchvision import transforms
from ts.torch_handler.image_classifier import ImageClassifier


class CustomImageClassifier(ImageClassifier):
    image_processing = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
