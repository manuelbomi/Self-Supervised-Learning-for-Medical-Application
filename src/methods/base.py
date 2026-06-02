"""
Base class for self-supervised learning methods.

Provides a common interface for SSL pretraining methods, handling shared
functionality like backbone management, optimizer configuration, and
checkpoint save/load. All SSL methods (SimCLR, DINO, MAE) inherit from
this base class.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)


@dataclass
class SSLConfig:
    """Configuration shared across all SSL methods."""

    # Backbone
    backbone: str = "vit_small"
    backbone_kwargs: Dict[str, Any] = field(default_factory=dict)
    embed_dim: int = 384
    img_size: int = 224
    in_channels: int = 1  # Medical images are typically single-channel

    # Optimizer
    optimizer: str = "adamw"
    base_lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 10
    min_lr: float = 1e-6

    # Training
    epochs: int = 200
    batch_size: int = 256
    fp16: bool = True

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 20
    resume_from: Optional[str] = None

    # Logging
    log_every: int = 50
    wandb_project: Optional[str] = None


class BaseSSLMethod(nn.Module, abc.ABC):
    """Abstract base class for self-supervised learning methods.

    Defines the common interface that all SSL methods must implement,
    including forward pass, loss computation, and representation extraction.

    Attributes:
        backbone: The encoder network producing representations.
        config: Shared SSL configuration.
    """

    def __init__(self, backbone: nn.Module, config: SSLConfig) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = config
        self._epoch: int = 0

    @abc.abstractmethod
    def forward(self, images: Tensor | List[Tensor], **kwargs: Any) -> Dict[str, Tensor]:
        """Forward pass through the SSL method.

        Args:
            images: Input image tensor(s). Shape depends on the method:
                - SimCLR: list of two tensors [view1, view2], each (B, C, H, W)
                - DINO: list of global + local crop tensors
                - MAE: single tensor (B, C, H, W)

        Returns:
            Dictionary containing at minimum:
                - "loss": scalar training loss
            Additional keys are method-specific (e.g., "logits", "targets").
        """
        ...

    @abc.abstractmethod
    def compute_loss(self, **kwargs: Any) -> Tensor:
        """Compute the SSL objective loss.

        Returns:
            Scalar loss tensor.
        """
        ...

    @torch.no_grad()
    def extract_features(self, images: Tensor) -> Tensor:
        """Extract representations from the backbone encoder.

        Used during evaluation (linear probe, k-NN, t-SNE). Runs in
        inference mode with no gradients.

        Args:
            images: Input images of shape (B, C, H, W).

        Returns:
            Feature tensor of shape (B, embed_dim).
        """
        self.backbone.eval()
        features = self.backbone(images)
        # Handle both ViT (returns class token) and CNN (returns pooled features)
        if isinstance(features, tuple):
            features = features[0]
        if features.dim() == 3:
            # ViT output: (B, num_patches+1, D) -> take CLS token
            features = features[:, 0]
        return features

    def get_learnable_params(self) -> List[Dict[str, Any]]:
        """Return parameter groups for the optimizer.

        Subclasses can override to define separate learning rates for
        different components (e.g., backbone vs. projection head).

        Returns:
            List of parameter group dictionaries.
        """
        regularized: List[Tensor] = []
        not_regularized: List[Tensor] = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name.endswith(".bias") or "norm" in name or "bn" in name:
                not_regularized.append(param)
            else:
                regularized.append(param)

        return [
            {"params": regularized, "weight_decay": self.config.weight_decay},
            {"params": not_regularized, "weight_decay": 0.0},
        ]

    def on_epoch_start(self, epoch: int) -> None:
        """Hook called at the beginning of each training epoch.

        Useful for updating schedules (e.g., EMA momentum in DINO,
        masking ratio in MAE).

        Args:
            epoch: Current epoch index (0-based).
        """
        self._epoch = epoch

    def on_step_end(self, step: int) -> None:
        """Hook called after each training step.

        Useful for EMA updates in DINO.

        Args:
            step: Global step index.
        """
        pass

    def save_checkpoint(self, path: Path, epoch: int, optimizer: Any = None) -> None:
        """Save model checkpoint to disk.

        Args:
            path: File path for the checkpoint.
            epoch: Current epoch number.
            optimizer: Optional optimizer state to include.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "method": self.__class__.__name__,
            "config": self.config.__dict__,
            "state_dict": self.state_dict(),
        }
        if optimizer is not None:
            checkpoint["optimizer"] = optimizer.state_dict()
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path} (epoch {epoch})")

    def load_checkpoint(
        self, path: Path, optimizer: Any = None, strict: bool = True
    ) -> int:
        """Load model checkpoint from disk.

        Args:
            path: Path to the checkpoint file.
            optimizer: Optional optimizer to restore state into.
            strict: Whether to enforce strict state dict matching.

        Returns:
            The epoch number stored in the checkpoint.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        self.load_state_dict(checkpoint["state_dict"], strict=strict)
        if optimizer is not None and "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint.get("epoch", 0)
        logger.info(f"Loaded checkpoint from {path} (epoch {epoch})")
        return epoch

    def extract_backbone_state_dict(self) -> Dict[str, Tensor]:
        """Extract only the backbone weights for downstream fine-tuning.

        Strips method-specific heads (projection, prediction, decoder) and
        returns only the encoder parameters.

        Returns:
            State dict containing only backbone parameters.
        """
        prefix = "backbone."
        return {
            k[len(prefix):]: v
            for k, v in self.state_dict().items()
            if k.startswith(prefix)
        }

    @property
    def device(self) -> torch.device:
        """Return the device of the model parameters."""
        return next(self.parameters()).device

    def __repr__(self) -> str:
        num_params = sum(p.numel() for p in self.parameters()) / 1e6
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        return (
            f"{self.__class__.__name__}(\n"
            f"  backbone={self.config.backbone},\n"
            f"  embed_dim={self.config.embed_dim},\n"
            f"  total_params={num_params:.1f}M,\n"
            f"  trainable_params={trainable:.1f}M\n"
            f")"
        )
