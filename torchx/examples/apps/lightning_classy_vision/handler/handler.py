# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchvision import transforms
# pyre-fixme[21]: Could not find module `ts.torch_handler.image_classifier`.
from ts.torch_handler.image_classifier import ImageClassifier


# pyre-fixme[11]: Annotation `ImageClassifier` is not defined as a type.
class CustomImageClassifier(ImageClassifier):
    image_processing = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
