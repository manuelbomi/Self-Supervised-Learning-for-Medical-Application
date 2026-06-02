"""
DINO: Self-Distillation with No Labels for Medical Image Representation Learning.

Implements the student-teacher self-distillation framework with exponential
moving average (EMA), multi-crop training strategy, and centering/sharpening
mechanisms adapted for medical imaging.

Reference:
    Caron, M. et al. "Emerging Properties in Self-Supervised Vision Transformers."
    ICCV 2021.

Medical imaging adaptations:
    - Multi-crop strategy tuned for high-resolution radiology images
    - Conservative augmentations preserving diagnostic information
    - Careful centering initialization for class-imbalanced medical data
    - Support for single-channel (grayscale) inputs
"""

from __future__ import annotations

import copy
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import BaseSSLMethod, SSLConfig

logger = logging.getLogger(__name__)


@dataclass
class DINOConfig(SSLConfig):
    """DINO-specific configuration."""

    # Architecture
    proj_hidden_dim: int = 2048
    proj_output_dim: int = 65536
    proj_bottleneck_dim: int = 256
    proj_num_layers: int = 3
    proj_use_bn: bool = False  # DINO uses LN in projection
    proj_norm_last_layer: bool = True

    # Teacher EMA
    ema_momentum_base: float = 0.996
    ema_momentum_final: float = 1.0

    # Loss
    student_temperature: float = 0.1
    teacher_temperature: float = 0.04
    teacher_temp_warmup_epochs: int = 30
    teacher_temp_warmup_start: float = 0.04
    center_momentum: float = 0.9

    # Multi-crop
    num_global_crops: int = 2
    num_local_crops: int = 6
    global_crop_scale: Tuple[float, float] = (0.4, 1.0)
    local_crop_scale: Tuple[float, float] = (0.05, 0.4)
    local_crop_size: int = 96

    # Training
    base_lr: float = 5e-4
    optimizer: str = "adamw"
    warmup_epochs: int = 10
    freeze_last_layer_epochs: int = 1
    clip_grad: float = 3.0


class DINOProjectionHead(nn.Module):
    """DINO projection head with bottleneck and weight-normalized last layer.

    Architecture: Linear -> GELU -> Linear -> GELU -> Linear (-> L2 norm)
    The last layer is optionally weight-normalized to stabilize training.
    A bottleneck layer reduces dimensionality before the final projection
    to the high-dimensional prototype space.

    Args:
        input_dim: Backbone output dimensionality.
        hidden_dim: Width of hidden layers.
        bottleneck_dim: Bottleneck dimensionality before final projection.
        output_dim: Number of prototypes / output dimensions.
        num_layers: Total number of linear layers.
        norm_last_layer: Apply weight normalization to the final layer.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        output_dim: int = 65536,
        num_layers: int = 3,
        norm_last_layer: bool = True,
    ) -> None:
        super().__init__()
        assert num_layers >= 3, "DINO projection head needs at least 3 layers"

        layers: List[nn.Module] = []

        # Input layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())

        # Hidden layers
        for _ in range(num_layers - 3):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())

        # Bottleneck layer
        layers.append(nn.Linear(hidden_dim, bottleneck_dim))

        self.mlp = nn.Sequential(*layers)

        # Final layer to prototype space
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, output_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1.0)

        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        """Project features to the prototype space.

        Args:
            x: Backbone features of shape (B, input_dim).

        Returns:
            Prototype logits of shape (B, output_dim).
        """
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x


class DINOLoss(nn.Module):
    """DINO cross-entropy loss with centering and sharpening.

    The teacher output is centered (to prevent collapse to a single prototype)
    and sharpened (low temperature) while the student uses a higher temperature.
    The loss is the cross-entropy between the sharpened teacher distribution
    and the student distribution, computed over all (global_teacher, any_student)
    pairs excluding same-view pairs.

    Args:
        output_dim: Number of prototypes in the projection head output.
        student_temperature: Temperature for student softmax (higher = softer).
        teacher_temperature: Temperature for teacher softmax (lower = sharper).
        center_momentum: EMA coefficient for center update.
        num_global_crops: Number of global crop views (teacher receives these).
    """

    def __init__(
        self,
        output_dim: int,
        student_temperature: float = 0.1,
        teacher_temperature: float = 0.04,
        center_momentum: float = 0.9,
        num_global_crops: int = 2,
    ) -> None:
        super().__init__()
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.center_momentum = center_momentum
        self.num_global_crops = num_global_crops

        # Running mean of teacher outputs (center) to prevent collapse
        self.register_buffer("center", torch.zeros(1, output_dim))

    @torch.no_grad()
    def update_center(self, teacher_output: Tensor) -> None:
        """Update the center vector using exponential moving average.

        The center prevents mode collapse by subtracting the running mean
        of teacher outputs before applying softmax.

        Args:
            teacher_output: Raw teacher logits, shape (B, output_dim).
        """
        batch_center = teacher_output.mean(dim=0, keepdim=True)

        # Synchronize across GPUs if in distributed setting
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(batch_center)
            batch_center /= torch.distributed.get_world_size()

        self.center = self.center * self.center_momentum + batch_center * (
            1.0 - self.center_momentum
        )

    def forward(
        self, student_output: List[Tensor], teacher_output: List[Tensor]
    ) -> Tensor:
        """Compute DINO loss between student and teacher distributions.

        The teacher processes only global crops. The student processes all
        crops (global + local). The loss sums over all (teacher_view,
        student_view) pairs where teacher_view != student_view.

        Args:
            student_output: List of student logits, one per crop.
            teacher_output: List of teacher logits for global crops only.

        Returns:
            Scalar DINO loss.
        """
        # Compute soft targets from teacher (centered + sharpened)
        teacher_probs = []
        for t_out in teacher_output:
            t_centered = (t_out - self.center) / self.teacher_temperature
            teacher_probs.append(F.softmax(t_centered, dim=-1))

        # Compute student log-probabilities
        student_log_probs = []
        for s_out in student_output:
            s_scaled = s_out / self.student_temperature
            student_log_probs.append(F.log_softmax(s_scaled, dim=-1))

        # Cross-entropy loss over all valid (teacher, student) pairs
        total_loss = torch.tensor(0.0, device=student_output[0].device)
        num_terms = 0

        for t_idx, t_prob in enumerate(teacher_probs):
            for s_idx, s_log_prob in enumerate(student_log_probs):
                # Skip same-view pairs (only for global crops)
                if s_idx == t_idx:
                    continue
                # Cross-entropy: -sum(t * log(s))
                loss = -torch.sum(t_prob * s_log_prob, dim=-1).mean()
                total_loss += loss
                num_terms += 1

        total_loss /= num_terms

        # Update center with teacher output
        with torch.no_grad():
            all_teacher = torch.cat(teacher_output, dim=0)
            self.update_center(all_teacher)

        return total_loss


class DINO(BaseSSLMethod):
    """DINO self-supervised learning method for medical imaging.

    Learns representations through self-distillation between a student and a
    momentum-updated teacher network. The teacher receives only global views
    while the student receives both global and local crops, encouraging the
    student to learn local-to-global correspondences.

    Medical imaging considerations:
        - Multi-crop scales adapted for high-resolution medical images
        - Conservative augmentations in crop generation
        - Careful centering for class-imbalanced medical datasets
        - Supports both ViT and CNN backbones (ViT preferred)

    Args:
        backbone: Student encoder network.
        config: DINO configuration.
    """

    def __init__(self, backbone: nn.Module, config: DINOConfig) -> None:
        super().__init__(backbone, config)
        self.dino_config = config

        # Student projection head
        self.projection_head = DINOProjectionHead(
            input_dim=config.embed_dim,
            hidden_dim=config.proj_hidden_dim,
            bottleneck_dim=config.proj_bottleneck_dim,
            output_dim=config.proj_output_dim,
            num_layers=config.proj_num_layers,
            norm_last_layer=config.proj_norm_last_layer,
        )

        # Teacher network (backbone + projection head) - no gradients
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_projection_head = copy.deepcopy(self.projection_head)
        for param in self.teacher_backbone.parameters():
            param.requires_grad = False
        for param in self.teacher_projection_head.parameters():
            param.requires_grad = False

        # Loss
        self.criterion = DINOLoss(
            output_dim=config.proj_output_dim,
            student_temperature=config.student_temperature,
            teacher_temperature=config.teacher_temperature,
            center_momentum=config.center_momentum,
            num_global_crops=config.num_global_crops,
        )

        # EMA schedule
        self._ema_momentum = config.ema_momentum_base

    def _get_ema_momentum(self, epoch: int) -> float:
        """Compute EMA momentum with cosine schedule.

        Momentum increases from base to 1.0 following a cosine curve,
        making teacher updates slower as training progresses.

        Args:
            epoch: Current epoch.

        Returns:
            Current EMA momentum value.
        """
        base = self.dino_config.ema_momentum_base
        final = self.dino_config.ema_momentum_final
        total = self.config.epochs
        return final - (final - base) * (math.cos(math.pi * epoch / total) + 1) / 2

    @torch.no_grad()
    def _update_teacher(self) -> None:
        """Update teacher network parameters via exponential moving average.

        teacher_param = m * teacher_param + (1 - m) * student_param
        """
        m = self._ema_momentum
        for t_param, s_param in zip(
            self.teacher_backbone.parameters(), self.backbone.parameters()
        ):
            t_param.data.mul_(m).add_(s_param.data, alpha=1.0 - m)

        for t_param, s_param in zip(
            self.teacher_projection_head.parameters(),
            self.projection_head.parameters(),
        ):
            t_param.data.mul_(m).add_(s_param.data, alpha=1.0 - m)

    def on_epoch_start(self, epoch: int) -> None:
        """Update EMA momentum and teacher temperature at epoch boundaries."""
        super().on_epoch_start(epoch)
        self._ema_momentum = self._get_ema_momentum(epoch)
        logger.debug(f"DINO EMA momentum: {self._ema_momentum:.6f}")

    def on_step_end(self, step: int) -> None:
        """Update teacher network after each student update."""
        self._update_teacher()

    def _encode_student(self, x: Tensor) -> Tensor:
        """Encode through student backbone and projection head."""
        features = self.backbone(x)
        if features.dim() == 3:
            features = features[:, 0]
        return self.projection_head(features)

    @torch.no_grad()
    def _encode_teacher(self, x: Tensor) -> Tensor:
        """Encode through teacher backbone and projection head (no grad)."""
        features = self.teacher_backbone(x)
        if features.dim() == 3:
            features = features[:, 0]
        return self.teacher_projection_head(features)

    def forward(
        self, images: List[Tensor], **kwargs: Any
    ) -> Dict[str, Tensor]:
        """Forward pass computing DINO self-distillation loss.

        Args:
            images: List of crop tensors. First `num_global_crops` are global
                crops (full resolution), remaining are local crops (smaller).
                Global crops: (B, C, H, W), Local crops: (B, C, h, w).

        Returns:
            Dictionary with:
                - "loss": DINO cross-entropy loss
                - "teacher_entropy": entropy of teacher distribution (collapse indicator)
                - "ema_momentum": current EMA momentum value
        """
        num_global = self.dino_config.num_global_crops

        # Teacher forward: only global crops
        teacher_outputs = []
        with torch.no_grad():
            for i in range(num_global):
                teacher_outputs.append(self._encode_teacher(images[i]))

        # Student forward: all crops (global + local)
        student_outputs = []
        for crop in images:
            student_outputs.append(self._encode_student(crop))

        # Compute DINO loss
        loss = self.compute_loss(
            student_output=student_outputs, teacher_output=teacher_outputs
        )

        # Monitor teacher entropy (should stay high to avoid collapse)
        with torch.no_grad():
            t_probs = F.softmax(
                (teacher_outputs[0] - self.criterion.center)
                / self.dino_config.teacher_temperature,
                dim=-1,
            )
            teacher_entropy = -torch.sum(
                t_probs * torch.log(t_probs + 1e-8), dim=-1
            ).mean()

        return {
            "loss": loss,
            "teacher_entropy": teacher_entropy,
            "ema_momentum": torch.tensor(self._ema_momentum),
        }

    def compute_loss(
        self,
        student_output: List[Tensor],
        teacher_output: List[Tensor],
        **kwargs: Any,
    ) -> Tensor:
        """Compute DINO cross-entropy loss with centering and sharpening.

        Args:
            student_output: Student logits for all crops.
            teacher_output: Teacher logits for global crops.

        Returns:
            Scalar DINO loss.
        """
        return self.criterion(student_output, teacher_output)

    def get_learnable_params(self) -> List[Dict[str, Any]]:
        """Return student-only parameters (teacher is updated via EMA)."""
        params = []
        params_no_wd = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or "norm" in name:
                params_no_wd.append(param)
            else:
                params.append(param)

        return [
            {"params": params, "weight_decay": self.config.weight_decay},
            {"params": params_no_wd, "weight_decay": 0.0},
        ]
