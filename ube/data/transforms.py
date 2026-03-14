from __future__ import annotations

from typing import Tuple

import torchvision.transforms as T

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_image_transform(image_size: int = 224) -> T.Compose:
    """Transforms for numpy array input (e.g. from NSD sample_dict).

    - Convert numpy to PIL.
    - Resize shorter side to image_size, then center-crop.
    - Convert to tensor in [0,1].
    - Normalize by ImageNet mean/std.
    """
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_image_transform_pil(image_size: int = 224) -> T.Compose:
    """Transforms for PIL Image input (e.g. loaded from Algonauts PNG files).

    - Resize shorter side to image_size, then center-crop.
    - Convert to tensor in [0,1].
    - Normalize by ImageNet mean/std.
    """
    return T.Compose(
        [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
