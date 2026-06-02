"""
Modified ResNet backbone for single-channel medical images.

Adapts the standard ResNet architecture for medical imaging:
    - Single-channel (grayscale) input support
    - Larger initial kernel for high-resolution medical images
    - Optional squeeze-and-excitation (SE) blocks for channel attention
    - Feature pyramid output for multi-scale representations

Reference:
    He, K. et al. "Deep Residual Learning for Image Recognition." CVPR 2016.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel recalibration.

    Adaptively recalibrates channel-wise feature responses by modelling
    interdependencies between channels. Particularly useful in medical
    imaging for learning which feature channels are relevant for different
    tissue types.

    Args:
        channels: Number of input channels.
        reduction: Reduction ratio for the bottleneck.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        mid_channels = max(channels // reduction, 8)
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        B, C, _, _ = x.shape
        scale = self.squeeze(x).view(B, C)
        scale = self.excitation(scale).view(B, C, 1, 1)
        return x * scale


class BasicBlock(nn.Module):
    """Basic residual block (two 3x3 convolutions).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        stride: Stride for the first convolution.
        downsample: Optional downsampling layer for the skip connection.
        use_se: Whether to use squeeze-and-excitation.
    """

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        self.se = SqueezeExcitation(out_channels) if use_se else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class Bottleneck(nn.Module):
    """Bottleneck residual block (1x1 -> 3x3 -> 1x1 convolutions).

    Args:
        in_channels: Number of input channels.
        out_channels: Number of intermediate channels (output is 4x this).
        stride: Stride for the 3x3 convolution.
        downsample: Optional downsampling for skip connection.
        use_se: Whether to include SE block.
    """

    expansion: int = 4

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        use_se: bool = False,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels, out_channels * self.expansion, 1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.se = (
            SqueezeExcitation(out_channels * self.expansion)
            if use_se
            else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ResNetMedical(nn.Module):
    """Modified ResNet for single-channel medical image analysis.

    Adaptations for medical imaging:
        - Single-channel input (grayscale mammograms, X-rays, CT slices)
        - Larger initial convolution kernel (7x7) for high-resolution inputs
        - Optional SE blocks for adaptive channel attention
        - Global average pooling to a fixed-size feature vector
        - Feature pyramid output for multi-scale analysis

    Args:
        block: Block type (BasicBlock or Bottleneck).
        layers: Number of blocks per stage [stage1, stage2, stage3, stage4].
        in_channels: Number of input channels (1 for grayscale medical images).
        base_width: Base channel width (64 for standard ResNet).
        use_se: Whether to use squeeze-and-excitation blocks.
        zero_init_residual: Zero-initialize the last BN in each block for
            better optimization landscape.
    """

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        in_channels: int = 1,
        base_width: int = 64,
        use_se: bool = False,
        zero_init_residual: bool = True,
    ) -> None:
        super().__init__()
        self.in_planes = base_width
        self.use_se = use_se

        # Initial convolution: adapted for single-channel medical images
        self.conv1 = nn.Conv2d(
            in_channels, base_width, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm2d(base_width)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual stages
        self.layer1 = self._make_layer(block, base_width, layers[0])
        self.layer2 = self._make_layer(block, base_width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_width * 8, layers[3], stride=2)

        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Output dimensionality
        self.embed_dim = base_width * 8 * block.expansion

        # Weight initialization
        self._init_weights(zero_init_residual)

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        num_blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Create a residual stage with the given number of blocks.

        Args:
            block: Block class to use.
            planes: Number of intermediate channels.
            num_blocks: Number of blocks in this stage.
            stride: Stride for the first block (downsampling).

        Returns:
            Sequential container of residual blocks.
        """
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_planes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [
            block(self.in_planes, planes, stride, downsample, use_se=self.use_se)
        ]
        self.in_planes = planes * block.expansion

        for _ in range(1, num_blocks):
            layers.append(
                block(self.in_planes, planes, use_se=self.use_se)
            )

        return nn.Sequential(*layers)

    def _init_weights(self, zero_init_residual: bool) -> None:
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # Zero-initialize last BN in each block for better training
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.zeros_(m.bn3.weight)
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.zeros_(m.bn2.weight)

    def forward_features(self, x: Tensor) -> Dict[str, Tensor]:
        """Extract multi-scale feature maps (feature pyramid).

        Returns features from each stage, useful for dense prediction
        tasks like segmentation.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Dictionary with keys 'stem', 'layer1' through 'layer4',
            and 'pooled' for the global average pooled features.
        """
        features: Dict[str, Tensor] = {}

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        features["stem"] = x

        x = self.maxpool(x)
        x = self.layer1(x)
        features["layer1"] = x

        x = self.layer2(x)
        features["layer2"] = x

        x = self.layer3(x)
        features["layer3"] = x

        x = self.layer4(x)
        features["layer4"] = x

        x = self.avgpool(x)
        features["pooled"] = x.flatten(1)

        return features

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass returning global average pooled features.

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Feature vector of shape (B, embed_dim).
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.flatten(1)


def resnet18_medical(in_channels: int = 1, use_se: bool = False, **kwargs) -> ResNetMedical:
    """ResNet-18 for medical imaging. ~11M parameters."""
    return ResNetMedical(BasicBlock, [2, 2, 2, 2], in_channels=in_channels, use_se=use_se, **kwargs)


def resnet34_medical(in_channels: int = 1, use_se: bool = False, **kwargs) -> ResNetMedical:
    """ResNet-34 for medical imaging. ~21M parameters."""
    return ResNetMedical(BasicBlock, [3, 4, 6, 3], in_channels=in_channels, use_se=use_se, **kwargs)


def resnet50_medical(in_channels: int = 1, use_se: bool = False, **kwargs) -> ResNetMedical:
    """ResNet-50 for medical imaging. ~23M parameters."""
    return ResNetMedical(Bottleneck, [3, 4, 6, 3], in_channels=in_channels, use_se=use_se, **kwargs)
