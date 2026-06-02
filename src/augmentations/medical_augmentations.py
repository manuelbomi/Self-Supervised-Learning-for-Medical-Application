"""
Medical image augmentation pipelines for self-supervised pretraining.

Provides carefully designed augmentation strategies for medical images that
preserve diagnostic information while creating sufficient view diversity for
contrastive and self-distillation learning.

Key design principles:
    - NO aggressive color jitter (would destroy intensity-based diagnostics)
    - Conservative geometric transforms (small rotations, limited flips)
    - Medical-specific augmentations (elastic deformation, artifact simulation)
    - Multi-crop strategy adapted for high-resolution radiology images
    - Intensity augmentations that respect the physics of medical imaging
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision import transforms as T
from torchvision.transforms import functional as TF


@dataclass
class MedicalAugConfig:
    """Configuration for medical image augmentations.

    Conservative defaults suited for mammography / chest X-ray data.
    """

    # Image properties
    img_size: int = 224
    in_channels: int = 1

    # Geometric augmentations (conservative for medical images)
    random_rotation_degrees: float = 15.0  # +/- degrees
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.0  # Off by default: anatomical orientation matters
    random_affine_translate: Tuple[float, float] = (0.05, 0.05)
    random_affine_scale: Tuple[float, float] = (0.95, 1.05)
    random_affine_shear: float = 5.0

    # Intensity augmentations
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    gamma_range: Tuple[float, float] = (0.8, 1.2)
    gaussian_noise_std: float = 0.02
    gaussian_blur_prob: float = 0.3
    gaussian_blur_sigma: Tuple[float, float] = (0.1, 2.0)

    # Medical-specific augmentations
    elastic_deformation_prob: float = 0.2
    elastic_alpha: float = 50.0
    elastic_sigma: float = 5.0
    random_erasing_prob: float = 0.1  # Simulate detector artifacts
    random_erasing_scale: Tuple[float, float] = (0.02, 0.1)
    clahe_prob: float = 0.3  # Contrast-limited adaptive histogram equalization
    window_level_jitter_prob: float = 0.3

    # Multi-crop (for DINO)
    num_global_crops: int = 2
    global_crop_scale: Tuple[float, float] = (0.4, 1.0)
    num_local_crops: int = 6
    local_crop_scale: Tuple[float, float] = (0.05, 0.4)
    local_crop_size: int = 96

    # Normalization (per-dataset, set during data loading)
    normalize_mean: float = 0.5
    normalize_std: float = 0.5


class GammaCorrection:
    """Random gamma correction for intensity augmentation.

    Simulates variation in exposure / display window settings commonly
    seen across different imaging equipment and clinical protocols.

    Args:
        gamma_range: Range (min, max) for random gamma value.
    """

    def __init__(self, gamma_range: Tuple[float, float] = (0.8, 1.2)) -> None:
        self.gamma_range = gamma_range

    def __call__(self, img: Tensor) -> Tensor:
        gamma = random.uniform(*self.gamma_range)
        # Clamp to [0, 1] for gamma correction
        img = img.clamp(0, 1)
        return img.pow(gamma)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(gamma_range={self.gamma_range})"


class GaussianNoise:
    """Add Gaussian noise to simulate detector noise in medical images.

    Medical images have characteristic noise patterns depending on the
    imaging modality (e.g., quantum noise in X-ray, thermal noise in MRI).

    Args:
        std: Standard deviation of the Gaussian noise.
    """

    def __init__(self, std: float = 0.02) -> None:
        self.std = std

    def __call__(self, img: Tensor) -> Tensor:
        noise = torch.randn_like(img) * self.std
        return (img + noise).clamp(0, 1)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(std={self.std})"


class ElasticDeformation:
    """Elastic deformation augmentation for medical images.

    Applies smooth random displacement fields to simulate tissue deformation
    and anatomical variability. Particularly useful for soft-tissue imaging
    (mammography, abdominal CT).

    Args:
        alpha: Magnitude of the displacement field.
        sigma: Smoothness of the displacement field (Gaussian filter sigma).
    """

    def __init__(self, alpha: float = 50.0, sigma: float = 5.0) -> None:
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, img: Tensor) -> Tensor:
        """Apply elastic deformation to a single image tensor.

        Args:
            img: Image tensor of shape (C, H, W).

        Returns:
            Deformed image tensor.
        """
        C, H, W = img.shape

        # Generate random displacement fields
        dx = torch.randn(1, 1, H, W) * self.alpha
        dy = torch.randn(1, 1, H, W) * self.alpha

        # Smooth displacement fields with Gaussian filter
        kernel_size = int(6 * self.sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        padding = kernel_size // 2

        # Create 1D Gaussian kernel
        x_coord = torch.arange(kernel_size).float() - padding
        kernel_1d = torch.exp(-x_coord**2 / (2 * self.sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()

        # Apply separable Gaussian filter
        kernel_h = kernel_1d.view(1, 1, -1, 1)
        kernel_w = kernel_1d.view(1, 1, 1, -1)

        dx = F.pad(dx, [0, 0, padding, padding], mode="reflect")
        dx = F.conv2d(dx, kernel_h)
        dx = F.pad(dx, [padding, padding, 0, 0], mode="reflect")
        dx = F.conv2d(dx, kernel_w)

        dy = F.pad(dy, [0, 0, padding, padding], mode="reflect")
        dy = F.conv2d(dy, kernel_h)
        dy = F.pad(dy, [padding, padding, 0, 0], mode="reflect")
        dy = F.conv2d(dy, kernel_w)

        # Create sampling grid
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H), torch.linspace(-1, 1, W), indexing="ij"
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # (1, H, W, 2)

        # Add displacement
        grid[..., 0] += dx.squeeze() * 2.0 / W
        grid[..., 1] += dy.squeeze() * 2.0 / H

        # Apply deformation
        img_4d = img.unsqueeze(0)
        deformed = F.grid_sample(
            img_4d, grid, mode="bilinear", padding_mode="border", align_corners=True
        )

        return deformed.squeeze(0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(alpha={self.alpha}, sigma={self.sigma})"


class WindowLevelJitter:
    """Simulate window/level adjustment variation in medical imaging.

    In clinical practice, radiologists adjust window width and level to
    optimize visualization of different tissue types. This augmentation
    simulates that variability.

    Args:
        width_range: Fractional range for window width jitter.
        level_range: Fractional range for window level jitter.
    """

    def __init__(
        self,
        width_range: Tuple[float, float] = (0.8, 1.2),
        level_range: Tuple[float, float] = (-0.1, 0.1),
    ) -> None:
        self.width_range = width_range
        self.level_range = level_range

    def __call__(self, img: Tensor) -> Tensor:
        width_factor = random.uniform(*self.width_range)
        level_shift = random.uniform(*self.level_range)

        center = 0.5 + level_shift
        half_width = 0.5 * width_factor

        lower = center - half_width
        upper = center + half_width

        img = (img - lower) / (upper - lower + 1e-8)
        return img.clamp(0, 1)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"width_range={self.width_range}, level_range={self.level_range})"
        )


def build_simclr_augmentation(config: MedicalAugConfig) -> T.Compose:
    """Build augmentation pipeline for SimCLR (two views).

    Creates a single transform that, when applied twice independently,
    produces two different augmented views for contrastive learning.

    Args:
        config: Augmentation configuration.

    Returns:
        Composed transform pipeline.
    """
    augmentations = [
        T.RandomResizedCrop(
            config.img_size,
            scale=config.global_crop_scale,
            interpolation=T.InterpolationMode.BICUBIC,
        ),
    ]

    # Geometric augmentations
    if config.horizontal_flip_prob > 0:
        augmentations.append(T.RandomHorizontalFlip(p=config.horizontal_flip_prob))
    if config.vertical_flip_prob > 0:
        augmentations.append(T.RandomVerticalFlip(p=config.vertical_flip_prob))
    if config.random_rotation_degrees > 0:
        augmentations.append(
            T.RandomRotation(config.random_rotation_degrees, fill=0)
        )

    # Intensity augmentations (conservative for medical images)
    augmentations.extend([
        T.RandomApply(
            [T.ColorJitter(
                brightness=(config.brightness_range[0] - 1, config.brightness_range[1] - 1),
                contrast=(config.contrast_range[0], config.contrast_range[1]),
            )],
            p=0.8,
        ),
        T.RandomApply([GammaCorrection(config.gamma_range)], p=0.5),
        T.RandomApply(
            [T.GaussianBlur(kernel_size=23, sigma=config.gaussian_blur_sigma)],
            p=config.gaussian_blur_prob,
        ),
        T.RandomApply([GaussianNoise(config.gaussian_noise_std)], p=0.3),
    ])

    # Medical-specific augmentations
    if config.elastic_deformation_prob > 0:
        augmentations.append(
            T.RandomApply(
                [ElasticDeformation(config.elastic_alpha, config.elastic_sigma)],
                p=config.elastic_deformation_prob,
            )
        )

    if config.random_erasing_prob > 0:
        augmentations.append(
            T.RandomErasing(
                p=config.random_erasing_prob,
                scale=config.random_erasing_scale,
                ratio=(0.3, 3.3),
                value=0,
            )
        )

    # Normalization
    augmentations.append(
        T.Normalize(mean=[config.normalize_mean], std=[config.normalize_std])
    )

    return T.Compose(augmentations)


def build_dino_multicrop(
    config: MedicalAugConfig,
) -> Tuple[List[T.Compose], List[T.Compose]]:
    """Build multi-crop augmentation pipeline for DINO.

    Creates separate transforms for global crops (large, high-resolution)
    and local crops (small, capturing fine details). The teacher receives
    only global crops; the student receives all crops.

    Args:
        config: Augmentation configuration.

    Returns:
        Tuple of (global_transforms, local_transforms).
    """
    # Base augmentations shared across all crops
    def _make_base_augmentations() -> List[Any]:
        return [
            T.RandomHorizontalFlip(p=config.horizontal_flip_prob),
            T.RandomApply(
                [T.ColorJitter(
                    brightness=(config.brightness_range[0] - 1, config.brightness_range[1] - 1),
                    contrast=(config.contrast_range[0], config.contrast_range[1]),
                )],
                p=0.8,
            ),
            T.RandomApply([GammaCorrection(config.gamma_range)], p=0.5),
            T.RandomApply([GaussianNoise(config.gaussian_noise_std)], p=0.2),
        ]

    # Global crop transforms
    global_transforms = []
    for _ in range(config.num_global_crops):
        transform = T.Compose([
            T.RandomResizedCrop(
                config.img_size,
                scale=config.global_crop_scale,
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            *_make_base_augmentations(),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=23, sigma=config.gaussian_blur_sigma)],
                p=0.5,
            ),
            T.RandomApply(
                [ElasticDeformation(config.elastic_alpha, config.elastic_sigma)],
                p=config.elastic_deformation_prob,
            ),
            T.Normalize(mean=[config.normalize_mean], std=[config.normalize_std]),
        ])
        global_transforms.append(transform)

    # Local crop transforms
    local_transforms = []
    for _ in range(config.num_local_crops):
        transform = T.Compose([
            T.RandomResizedCrop(
                config.local_crop_size,
                scale=config.local_crop_scale,
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            *_make_base_augmentations(),
            T.RandomApply(
                [T.GaussianBlur(kernel_size=23, sigma=config.gaussian_blur_sigma)],
                p=0.3,
            ),
            T.Normalize(mean=[config.normalize_mean], std=[config.normalize_std]),
        ])
        local_transforms.append(transform)

    return global_transforms, local_transforms


def build_mae_augmentation(config: MedicalAugConfig) -> T.Compose:
    """Build minimal augmentation pipeline for MAE.

    MAE relies on masking as the primary pretext task, so only light
    augmentations are needed. Heavy augmentations can hurt MAE performance.

    Args:
        config: Augmentation configuration.

    Returns:
        Composed transform pipeline.
    """
    return T.Compose([
        T.RandomResizedCrop(
            config.img_size,
            scale=(0.5, 1.0),
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(p=config.horizontal_flip_prob),
        T.Normalize(mean=[config.normalize_mean], std=[config.normalize_std]),
    ])


def build_eval_augmentation(config: MedicalAugConfig) -> T.Compose:
    """Build deterministic augmentation pipeline for evaluation.

    No random augmentations -- only resize, center crop, and normalize.

    Args:
        config: Augmentation configuration.

    Returns:
        Deterministic transform pipeline.
    """
    return T.Compose([
        T.Resize(
            int(config.img_size * 256 / 224),
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.CenterCrop(config.img_size),
        T.Normalize(mean=[config.normalize_mean], std=[config.normalize_std]),
    ])


class MultiCropWrapper:
    """Wraps an image to apply multiple crop transforms for DINO.

    Applies all global and local crop transforms to the same source image,
    returning a list of cropped tensors.

    Args:
        global_transforms: List of transforms for global crops.
        local_transforms: List of transforms for local crops.
    """

    def __init__(
        self,
        global_transforms: List[T.Compose],
        local_transforms: List[T.Compose],
    ) -> None:
        self.global_transforms = global_transforms
        self.local_transforms = local_transforms

    def __call__(self, img: Tensor) -> List[Tensor]:
        """Apply all crop transforms to the input image.

        Args:
            img: Input image tensor (C, H, W).

        Returns:
            List of cropped tensors: [global_1, global_2, local_1, ..., local_N].
        """
        crops = []
        for transform in self.global_transforms:
            crops.append(transform(img))
        for transform in self.local_transforms:
            crops.append(transform(img))
        return crops

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"n_global={len(self.global_transforms)}, "
            f"n_local={len(self.local_transforms)})"
        )


class DualViewTransform:
    """Applies the same base transform independently twice for SimCLR.

    Args:
        transform: Base augmentation transform.
    """

    def __init__(self, transform: T.Compose) -> None:
        self.transform = transform

    def __call__(self, img: Tensor) -> List[Tensor]:
        """Generate two augmented views of the input.

        Args:
            img: Input image tensor (C, H, W).

        Returns:
            List of two augmented views [view1, view2].
        """
        return [self.transform(img), self.transform(img)]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(transform={self.transform})"
