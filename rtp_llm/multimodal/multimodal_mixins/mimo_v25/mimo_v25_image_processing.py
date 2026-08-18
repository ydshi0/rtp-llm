"""MiMo V2.5 image preprocessing.

Mirrors the official MiMo-VL processor: a single bilinear resize over raw
0-255 float pixels followed by ImageNet mean/std standardisation, then the
Qwen-style patch flattening.

Qwen2VLImageProcessor is deliberately not reused here. It resizes with PIL
bicubic and normalises with the CLIP mean/std, so it produces different pixel
values for the same image and would fail a layer-by-layer golden comparison
even though its patch layout is identical.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .modeling_mimo_vit import MiMoVisionConfig

# 0-255 ImageNet statistics, matching the official MiMo processor.
PIXEL_MEAN = [123.675, 116.28, 103.53]
PIXEL_STD = [58.395, 57.12, 57.375]

MAX_ASPECT_RATIO = 200


def smart_resize(
    height: int, width: int, factor: int, min_pixels: int, max_pixels: int
) -> Tuple[int, int]:
    """Official MiMo resize policy: multiple of `factor`, within pixel bounds."""
    if min(height, width) < factor:
        if height < width:
            height, width = factor, int(width * factor / height)
        else:
            width, height = factor, int(height * factor / width)
    elif max(height, width) / min(height, width) > MAX_ASPECT_RATIO:
        raise ValueError(
            f"MiMo VIT rejects aspect ratio above {MAX_ASPECT_RATIO}: "
            f"{height}x{width}"
        )
    resized_height = round(height / factor) * factor
    resized_width = round(width / factor) * factor
    if resized_height * resized_width > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        resized_height = math.floor(height / beta / factor) * factor
        resized_width = math.floor(width / beta / factor) * factor
    elif resized_height * resized_width < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        resized_height = math.ceil(height * beta / factor) * factor
        resized_width = math.ceil(width * beta / factor) * factor
    return int(resized_height), int(resized_width)


class MiMoV25ImageProcessor:
    def __init__(
        self,
        patch_size: int,
        spatial_merge_size: int,
        temporal_patch_size: int,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ):
        self.patch_size = patch_size
        self.merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.size_factor = patch_size * spatial_merge_size
        unit = self.size_factor * self.size_factor
        self.min_pixels = min_pixels or 4 * unit
        self.max_pixels = max_pixels or 4096 * unit

    @classmethod
    def from_vision_config(
        cls,
        config: MiMoVisionConfig,
        processor_config: Optional[Dict] = None,
    ) -> "MiMoV25ImageProcessor":
        processor_config = processor_config or {}
        return cls(
            patch_size=config.patch_size,
            spatial_merge_size=config.spatial_merge_size,
            temporal_patch_size=config.temporal_patch_size,
            min_pixels=processor_config.get("image_min_pixels"),
            max_pixels=processor_config.get("image_max_pixels"),
        )

    def resize_and_normalize(
        self,
        image: Image.Image,
        min_pixels: int,
        max_pixels: int,
        nominal_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """Return standardised pixels as [1, C, resized_height, resized_width].

        `nominal_size` is the (height, width) fed to the resize policy when the
        request overrides it; the interpolation still starts from the original
        pixels so the image is only resampled once.
        """
        image = image.convert("RGB")
        width, height = image.size
        pixels = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        nominal_height, nominal_width = nominal_size or (height, width)
        resized_height, resized_width = smart_resize(
            nominal_height,
            nominal_width,
            self.size_factor,
            min_pixels,
            max_pixels,
        )
        resized = F.interpolate(
            pixels.unsqueeze(0),
            (resized_height, resized_width),
            mode="bilinear",
            align_corners=False,
        )
        mean = torch.tensor(PIXEL_MEAN).view(1, -1, 1, 1)
        std = torch.tensor(PIXEL_STD).view(1, -1, 1, 1)
        return (resized - mean) / std

    def flatten_patches(
        self, pixels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """[1, C, H, W] -> ([grid_t * grid_h * grid_w, patch_dim], grid_thw)."""
        # A still image is replicated across the temporal patch, giving grid_t 1.
        patches = pixels.repeat(self.temporal_patch_size, 1, 1, 1)
        channels = patches.shape[1]
        height, width = patches.shape[-2:]
        if height % self.size_factor or width % self.size_factor:
            raise ValueError(
                f"resized image {height}x{width} is not a multiple of "
                f"patch_size * spatial_merge_size ({self.size_factor})"
            )
        grid_t = patches.shape[0] // self.temporal_patch_size
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        patches = (
            patches.contiguous()
            .view(
                grid_t,
                self.temporal_patch_size,
                channels,
                grid_h // self.merge_size,
                self.merge_size,
                self.patch_size,
                grid_w // self.merge_size,
                self.merge_size,
                self.patch_size,
            )
            .permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
            .contiguous()
            .view(
                grid_t * grid_h * grid_w,
                channels
                * self.temporal_patch_size
                * self.patch_size
                * self.patch_size,
            )
        )
        grid_thw = torch.tensor([[grid_t, grid_h, grid_w]], dtype=torch.long)
        return patches, grid_thw

    def __call__(
        self,
        image: Image.Image,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        nominal_size: Optional[Tuple[int, int]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pixels = self.resize_and_normalize(
            image,
            min_pixels if min_pixels is not None else self.min_pixels,
            max_pixels if max_pixels is not None else self.max_pixels,
            nominal_size=nominal_size,
        )
        return self.flatten_patches(pixels)
