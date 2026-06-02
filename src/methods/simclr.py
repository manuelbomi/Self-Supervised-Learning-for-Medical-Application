"""
SimCLR: A Simple Framework for Contrastive Learning of Visual Representations.

Adapted for medical imaging with domain-specific augmentations and single-channel
support. Implements the NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss
with a learnable projection head.

Reference:
    Chen, T. et al. "A Simple Framework for Contrastive Learning of Visual
    Representations." ICML 2020.

Key medical imaging adaptations:
    - Conservative augmentation pipeline (no aggressive color distortion)
    - Single-channel input support (grayscale mammograms, X-rays)
    - LARS optimizer support for large-batch training stability
    - Gradient accumulation for memory-constrained medical image sizes
"""

from __future__ import annotations

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
class SimCLRConfig(SSLConfig):
    """SimCLR-specific configuration."""

    # Projection head
    proj_hidden_dim: int = 2048
    proj_output_dim: int = 128
    proj_num_layers: int = 2
    proj_use_bn: bool = True

    # NT-Xent loss
    temperature: float = 0.07
    learnable_temperature: bool = False

    # Training
    base_lr: float = 0.3  # SimCLR uses higher LR with LARS
    optimizer: str = "lars"
    warmup_epochs: int = 10
    batch_size: int = 512

    # Augmentation
    use_medical_augmentations: bool = True


class ProjectionHead(nn.Module):
    """Non-linear projection head g(.) mapping representations to the space
    where the contrastive loss is applied.

    Architecture: Linear -> BN -> ReLU -> Linear [-> BN -> ReLU -> Linear]
    The final layer has no activation, producing the embeddings used for
    NT-Xent loss computation.

    Args:
        input_dim: Dimensionality of backbone output.
        hidden_dim: Hidden layer width.
        output_dim: Projection output dimensionality.
        num_layers: Number of linear layers (2 or 3).
        use_bn: Whether to use batch normalization.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 2048,
        output_dim: int = 128,
        num_layers: int = 2,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        assert num_layers >= 2, "Projection head must have at least 2 layers"

        layers: List[nn.Module] = []

        # First layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU(inplace=True))

        # Intermediate layers
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        # Final layer (no activation, optional BN)
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Project backbone features to contrastive embedding space.

        Args:
            x: Feature tensor of shape (B, input_dim).

        Returns:
            Projected embeddings of shape (B, output_dim).
        """
        return self.mlp(x)


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross-Entropy Loss (NT-Xent).

    For a batch of N pairs, creates a 2N x 2N similarity matrix. The positive
    pair for sample i is its augmented counterpart. All other 2(N-1) samples
    are treated as negatives. The loss is the negative log-softmax of the
    positive similarity over all similarities (excluding self-similarity).

    Args:
        temperature: Scaling factor for logits. Lower values create sharper
            distributions, emphasizing hard negatives. Default 0.07 works well
            for medical imaging where subtle differences matter.
        learnable: Whether temperature is a learnable parameter.
    """

    def __init__(self, temperature: float = 0.07, learnable: bool = False) -> None:
        super().__init__()
        if learnable:
            self.temperature = nn.Parameter(torch.tensor(math.log(1.0 / temperature)))
        else:
            self.register_buffer("temperature", torch.tensor(temperature))
        self.learnable = learnable

    def _get_temperature(self) -> Tensor:
        """Return the effective temperature value."""
        if self.learnable:
            return torch.exp(-self.temperature)  # Stored as log(1/t)
        return self.temperature

    def forward(self, z_i: Tensor, z_j: Tensor) -> Tensor:
        """Compute NT-Xent loss for a batch of positive pairs.

        Args:
            z_i: Projected features from view 1, shape (B, D).
            z_j: Projected features from view 2, shape (B, D).

        Returns:
            Scalar NT-Xent loss.
        """
        batch_size = z_i.shape[0]
        device = z_i.device
        temperature = self._get_temperature()

        # L2 normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate: [z_i; z_j] -> (2B, D)
        z = torch.cat([z_i, z_j], dim=0)

        # Compute full pairwise cosine similarity matrix: (2B, 2B)
        sim_matrix = torch.mm(z, z.t()) / temperature

        # Mask out self-similarity on the diagonal
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        sim_matrix.masked_fill_(mask, -1e9)

        # For each sample i in [0, 2B), its positive is at index (i + B) % 2B
        # Positive pairs: (0, B), (1, B+1), ..., (B-1, 2B-1), (B, 0), ...
        pos_indices = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0, batch_size, device=device),
        ])

        # Gather positive similarities
        positives = sim_matrix[torch.arange(2 * batch_size, device=device), pos_indices]

        # NT-Xent: -log(exp(pos) / sum(exp(all non-self)))
        # Equivalent to cross-entropy with positive index as target
        loss = -positives + torch.logsumexp(sim_matrix, dim=1)

        return loss.mean()


class SimCLR(BaseSSLMethod):
    """SimCLR self-supervised learning method for medical imaging.

    Learns representations by maximizing agreement between differently augmented
    views of the same image through a contrastive loss in a learned projection
    space. The projection head is discarded after pretraining; only the backbone
    encoder is used for downstream tasks.

    Medical imaging considerations:
        - Uses conservative augmentations (no aggressive color jitter)
        - Lower temperature (0.07) to capture subtle radiological differences
        - LARS optimizer for stable large-batch training
        - Supports single-channel (grayscale) inputs

    Args:
        backbone: Encoder network (ResNet or ViT).
        config: SimCLR configuration.
    """

    def __init__(self, backbone: nn.Module, config: SimCLRConfig) -> None:
        super().__init__(backbone, config)
        self.simclr_config = config

        # Projection head: maps backbone features to contrastive space
        self.projection_head = ProjectionHead(
            input_dim=config.embed_dim,
            hidden_dim=config.proj_hidden_dim,
            output_dim=config.proj_output_dim,
            num_layers=config.proj_num_layers,
            use_bn=config.proj_use_bn,
        )

        # Contrastive loss
        self.criterion = NTXentLoss(
            temperature=config.temperature,
            learnable=config.learnable_temperature,
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize projection head weights with proper scaling."""
        for module in self.projection_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _encode_and_project(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Encode input through backbone and project to contrastive space.

        Args:
            x: Input images of shape (B, C, H, W).

        Returns:
            Tuple of (backbone_features, projected_embeddings).
        """
        features = self.backbone(x)
        # Handle ViT output: (B, num_patches+1, D) -> CLS token
        if features.dim() == 3:
            features = features[:, 0]
        projections = self.projection_head(features)
        return features, projections

    def forward(
        self, images: List[Tensor], **kwargs: Any
    ) -> Dict[str, Tensor]:
        """Forward pass computing SimCLR contrastive loss.

        Args:
            images: List of two tensors [view1, view2], each (B, C, H, W),
                representing two augmented views of the same batch of images.

        Returns:
            Dictionary with:
                - "loss": NT-Xent contrastive loss
                - "z1", "z2": projected embeddings (for monitoring)
                - "avg_pos_sim": average positive pair similarity
                - "avg_neg_sim": average negative pair similarity
        """
        assert len(images) == 2, "SimCLR requires exactly 2 views"
        view1, view2 = images

        # Encode and project both views
        _, z1 = self._encode_and_project(view1)
        _, z2 = self._encode_and_project(view2)

        # Compute contrastive loss
        loss = self.compute_loss(z1=z1, z2=z2)

        # Compute monitoring metrics
        with torch.no_grad():
            z1_norm = F.normalize(z1, dim=1)
            z2_norm = F.normalize(z2, dim=1)
            avg_pos_sim = (z1_norm * z2_norm).sum(dim=1).mean()
            # Negative similarity: average of off-diagonal elements
            sim_matrix = torch.mm(z1_norm, z2_norm.t())
            mask = ~torch.eye(z1.shape[0], dtype=torch.bool, device=z1.device)
            avg_neg_sim = sim_matrix[mask].mean()

        return {
            "loss": loss,
            "z1": z1.detach(),
            "z2": z2.detach(),
            "avg_pos_sim": avg_pos_sim,
            "avg_neg_sim": avg_neg_sim,
        }

    def compute_loss(self, z1: Tensor, z2: Tensor, **kwargs: Any) -> Tensor:
        """Compute NT-Xent contrastive loss.

        Args:
            z1: Projected embeddings from view 1, shape (B, proj_dim).
            z2: Projected embeddings from view 2, shape (B, proj_dim).

        Returns:
            Scalar NT-Xent loss.
        """
        return self.criterion(z1, z2)

    def get_learnable_params(self) -> List[Dict[str, Any]]:
        """Return parameter groups with separate handling for projection head.

        The projection head can use a higher learning rate than the backbone
        since it is trained from scratch.
        """
        backbone_params = []
        backbone_params_no_wd = []
        proj_params = []
        proj_params_no_wd = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            is_proj = name.startswith("projection_head") or name.startswith("criterion")
            no_wd = name.endswith(".bias") or "norm" in name or "bn" in name

            if is_proj:
                (proj_params_no_wd if no_wd else proj_params).append(param)
            else:
                (backbone_params_no_wd if no_wd else backbone_params).append(param)

        return [
            {"params": backbone_params, "weight_decay": self.config.weight_decay},
            {"params": backbone_params_no_wd, "weight_decay": 0.0},
            {"params": proj_params, "weight_decay": self.config.weight_decay, "lr_scale": 1.0},
            {"params": proj_params_no_wd, "weight_decay": 0.0, "lr_scale": 1.0},
        ]
