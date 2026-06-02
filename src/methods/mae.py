"""
MAE: Masked Autoencoders Are Scalable Vision Learners for Medical Imaging.

Implements the Masked Autoencoder framework with a ViT encoder that processes
only visible (unmasked) patches and a lightweight decoder that reconstructs
masked patches. Adapted for medical imaging with domain-specific masking
strategies and single-channel reconstruction.

Reference:
    He, K. et al. "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022.

Medical imaging adaptations:
    - Anatomically-aware masking strategies (block masking, grid masking)
    - Single-channel reconstruction loss (MSE in pixel space)
    - Optional frequency-weighted loss for structure-sensitive reconstruction
    - Configurable masking ratio (default 75% for medical images)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .base import BaseSSLMethod, SSLConfig

logger = logging.getLogger(__name__)


@dataclass
class MAEConfig(SSLConfig):
    """MAE-specific configuration."""

    # Masking
    mask_ratio: float = 0.75
    masking_strategy: str = "random"  # "random", "block", "grid"
    block_aspect_ratio: Tuple[float, float] = (0.3, 3.3)

    # Decoder
    decoder_embed_dim: int = 256
    decoder_depth: int = 4
    decoder_num_heads: int = 8

    # Loss
    norm_pix_loss: bool = True  # Normalize pixel values per patch
    loss_on_masked_only: bool = True  # Compute loss only on masked patches
    frequency_weight: float = 0.0  # Weight for frequency-domain loss

    # Encoder (inherited from SSLConfig)
    patch_size: int = 16

    # Training
    base_lr: float = 1.5e-4
    optimizer: str = "adamw"
    warmup_epochs: int = 40
    weight_decay: float = 0.05


class PatchEmbed(nn.Module):
    """Convert image to sequence of patch embeddings.

    Args:
        img_size: Input image size (assumed square).
        patch_size: Size of each patch.
        in_channels: Number of input channels.
        embed_dim: Embedding dimensionality.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: Tensor) -> Tensor:
        """Embed patches from input image.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim).
        """
        # (B, C, H, W) -> (B, D, H/P, W/P) -> (B, D, N) -> (B, N, D)
        return self.proj(x).flatten(2).transpose(1, 2)


class MAEDecoder(nn.Module):
    """Lightweight transformer decoder for MAE reconstruction.

    Receives encoded visible patch tokens plus learnable mask tokens,
    and reconstructs the original pixel values for all patches.

    Args:
        num_patches: Total number of patches in the image.
        encoder_embed_dim: Dimensionality of encoder output.
        decoder_embed_dim: Dimensionality of decoder hidden state.
        decoder_depth: Number of transformer decoder blocks.
        decoder_num_heads: Number of attention heads.
        patch_size: Spatial size of each patch.
        in_channels: Number of image channels (1 for medical grayscale).
    """

    def __init__(
        self,
        num_patches: int,
        encoder_embed_dim: int = 384,
        decoder_embed_dim: int = 256,
        decoder_depth: int = 4,
        decoder_num_heads: int = 8,
        patch_size: int = 16,
        in_channels: int = 1,
    ) -> None:
        super().__init__()

        self.num_patches = num_patches
        self.patch_size = patch_size
        self.in_channels = in_channels

        # Project encoder dim to decoder dim
        self.decoder_embed = nn.Linear(encoder_embed_dim, decoder_embed_dim)

        # Learnable mask token: placeholder for masked positions
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # Positional embeddings for decoder (fixed sinusoidal)
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False
        )
        self._init_pos_embed()

        # Transformer blocks
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=decoder_embed_dim,
                nhead=decoder_num_heads,
                dim_feedforward=decoder_embed_dim * 4,
                dropout=0.0,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(decoder_depth)
        ])

        self.norm = nn.LayerNorm(decoder_embed_dim)

        # Predict pixel values for each patch
        self.head = nn.Linear(decoder_embed_dim, patch_size * patch_size * in_channels)

    def _init_pos_embed(self) -> None:
        """Initialize 2D sinusoidal positional embeddings."""
        grid_size = int(self.num_patches**0.5)
        embed_dim = self.decoder_pos_embed.shape[-1]

        grid_h = torch.arange(grid_size, dtype=torch.float32)
        grid_w = torch.arange(grid_size, dtype=torch.float32)
        grid = torch.meshgrid(grid_h, grid_w, indexing="ij")
        grid = torch.stack(grid, dim=0).reshape(2, -1)  # (2, N)

        half_dim = embed_dim // 4
        omega = 1.0 / (10000.0 ** (torch.arange(half_dim).float() / half_dim))

        pos_embed = torch.zeros(self.num_patches, embed_dim)
        for i in range(2):  # height and width
            out = grid[i].unsqueeze(1) * omega.unsqueeze(0)  # (N, half_dim)
            pos_embed[:, i * (embed_dim // 2): i * (embed_dim // 2) + half_dim] = torch.sin(out)
            pos_embed[:, i * (embed_dim // 2) + half_dim: (i + 1) * (embed_dim // 2)] = torch.cos(out)

        self.decoder_pos_embed.data.copy_(pos_embed.unsqueeze(0))

    def forward(
        self, x: Tensor, ids_restore: Tensor, mask: Tensor
    ) -> Tensor:
        """Decode encoded visible patches and predict reconstruction.

        Args:
            x: Encoded visible patch tokens, shape (B, num_visible, encoder_dim).
            ids_restore: Indices to unshuffle tokens back to original order,
                shape (B, num_patches).
            mask: Binary mask indicating masked patches (1 = masked),
                shape (B, num_patches).

        Returns:
            Reconstructed patch pixels, shape (B, num_patches, patch_size^2 * C).
        """
        B, N_vis, _ = x.shape

        # Project to decoder dimension
        x = self.decoder_embed(x)

        # Append mask tokens for masked positions
        num_masked = self.num_patches - N_vis
        mask_tokens = self.mask_token.expand(B, num_masked, -1)
        x = torch.cat([x, mask_tokens], dim=1)

        # Unshuffle to restore original spatial order
        x = torch.gather(
            x, dim=1, index=ids_restore.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        )

        # Add positional embeddings
        x = x + self.decoder_pos_embed

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # Predict pixel values
        x = self.head(x)

        return x


class MAE(BaseSSLMethod):
    """Masked Autoencoder for medical image self-supervised learning.

    Masks a high proportion (75%) of image patches, encodes only the visible
    patches with a ViT encoder, and reconstructs the masked patches with a
    lightweight decoder. This asymmetric design is computationally efficient
    and learns strong representations.

    Medical imaging considerations:
        - Single-channel reconstruction (grayscale)
        - Optional per-patch pixel normalization for intensity invariance
        - Block masking strategy for learning structural priors
        - Frequency-weighted loss for edge/structure preservation

    Args:
        backbone: ViT encoder that supports partial patch input.
        config: MAE configuration.
    """

    def __init__(self, backbone: nn.Module, config: MAEConfig) -> None:
        super().__init__(backbone, config)
        self.mae_config = config

        # Patch embedding for tokenization
        self.patch_embed = PatchEmbed(
            img_size=config.img_size,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
            embed_dim=config.embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        # Positional embedding for encoder (learnable)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, config.embed_dim)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Decoder
        self.decoder = MAEDecoder(
            num_patches=num_patches,
            encoder_embed_dim=config.embed_dim,
            decoder_embed_dim=config.decoder_embed_dim,
            decoder_depth=config.decoder_depth,
            decoder_num_heads=config.decoder_num_heads,
            patch_size=config.patch_size,
            in_channels=config.in_channels,
        )

    def _random_masking(
        self, x: Tensor, mask_ratio: float
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Perform per-sample random masking by shuffling patch indices.

        Args:
            x: Patch embeddings of shape (B, N, D).
            mask_ratio: Fraction of patches to mask.

        Returns:
            x_visible: Visible patch embeddings (B, N_vis, D).
            mask: Binary mask (B, N), 1 = masked.
            ids_restore: Indices to restore original order (B, N).
        """
        B, N, D = x.shape
        num_keep = int(N * (1 - mask_ratio))

        # Random noise for sorting
        noise = torch.rand(B, N, device=x.device)

        # Sort noise: small values -> kept, large values -> masked
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # Keep first num_keep tokens
        ids_keep = ids_shuffle[:, :num_keep]
        x_visible = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        # Generate binary mask: 0 = keep, 1 = mask
        mask = torch.ones(B, N, device=x.device)
        mask[:, :num_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_visible, mask, ids_restore

    def _block_masking(
        self, x: Tensor, mask_ratio: float
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Block-wise masking that masks contiguous rectangular regions.

        This encourages learning of spatial structure, which is important
        for medical images where pathology has spatial extent.

        Args:
            x: Patch embeddings of shape (B, N, D).
            mask_ratio: Target fraction of patches to mask.

        Returns:
            Same as _random_masking.
        """
        B, N, D = x.shape
        grid_size = int(N**0.5)
        num_mask = int(N * mask_ratio)

        mask = torch.zeros(B, N, device=x.device)

        for b in range(B):
            masked_count = 0
            while masked_count < num_mask:
                # Random block dimensions
                ar = torch.empty(1).uniform_(*self.mae_config.block_aspect_ratio).item()
                block_area = min(int(torch.empty(1).uniform_(1, num_mask - masked_count + 1).item()), num_mask - masked_count)
                block_h = max(1, min(int(math.sqrt(block_area * ar)), grid_size))
                block_w = max(1, min(block_area // block_h, grid_size))

                # Random position
                top = torch.randint(0, grid_size - block_h + 1, (1,)).item()
                left = torch.randint(0, grid_size - block_w + 1, (1,)).item()

                for r in range(top, top + block_h):
                    for c in range(left, left + block_w):
                        idx = r * grid_size + c
                        if mask[b, idx] == 0:
                            mask[b, idx] = 1
                            masked_count += 1
                            if masked_count >= num_mask:
                                break
                    if masked_count >= num_mask:
                        break

        # Derive keep indices from mask
        ids_shuffle = torch.argsort(mask, dim=1)  # 0s first (kept)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        num_keep = N - num_mask
        ids_keep = ids_shuffle[:, :num_keep]
        x_visible = torch.gather(
            x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )

        return x_visible, mask, ids_restore

    def _patchify(self, imgs: Tensor) -> Tensor:
        """Convert images to patch sequences for loss computation.

        Args:
            imgs: Input images (B, C, H, W).

        Returns:
            Patches of shape (B, num_patches, patch_size^2 * C).
        """
        p = self.mae_config.patch_size
        c = self.mae_config.in_channels
        h = w = imgs.shape[2] // p

        x = imgs.reshape(imgs.shape[0], c, h, p, w, p)
        x = x.permute(0, 2, 4, 3, 5, 1)  # (B, h, w, p, p, C)
        x = x.reshape(imgs.shape[0], h * w, p * p * c)  # (B, N, p*p*C)
        return x

    def forward(
        self, images: Tensor | List[Tensor], **kwargs: Any
    ) -> Dict[str, Tensor]:
        """Forward pass: mask, encode visible patches, decode, compute loss.

        Args:
            images: Input images (B, C, H, W) or single-element list.

        Returns:
            Dictionary with:
                - "loss": reconstruction loss (MSE)
                - "pred": predicted patch pixels
                - "mask": binary mask (for visualization)
                - "reconstruction_error": per-patch MSE (for analysis)
        """
        if isinstance(images, list):
            images = images[0]

        # Patchify input
        patches = self.patch_embed(images)

        # Add positional embeddings
        patches = patches + self.pos_embed

        # Apply masking strategy
        if self.mae_config.masking_strategy == "block":
            x_visible, mask, ids_restore = self._block_masking(
                patches, self.mae_config.mask_ratio
            )
        else:
            x_visible, mask, ids_restore = self._random_masking(
                patches, self.mae_config.mask_ratio
            )

        # Encode visible patches through backbone
        # Note: backbone must handle variable-length sequences
        encoded = self.backbone(x_visible, is_partial=True)
        if isinstance(encoded, tuple):
            encoded = encoded[0]

        # Decode: reconstruct all patches
        pred = self.decoder(encoded, ids_restore, mask)

        # Compute loss
        loss = self.compute_loss(images=images, pred=pred, mask=mask)

        # Per-patch reconstruction error (for analysis)
        with torch.no_grad():
            target = self._patchify(images)
            per_patch_mse = ((pred - target) ** 2).mean(dim=-1)

        return {
            "loss": loss,
            "pred": pred.detach(),
            "mask": mask,
            "reconstruction_error": per_patch_mse,
        }

    def compute_loss(
        self,
        images: Tensor,
        pred: Tensor,
        mask: Tensor,
        **kwargs: Any,
    ) -> Tensor:
        """Compute reconstruction loss (MSE) on masked patches.

        Optionally normalizes each patch by its mean and variance before
        computing the loss, which improves representation quality.

        Args:
            images: Original images (B, C, H, W).
            pred: Predicted patch pixels (B, N, patch_size^2 * C).
            mask: Binary mask (B, N), 1 = masked.

        Returns:
            Scalar reconstruction loss.
        """
        target = self._patchify(images)

        if self.mae_config.norm_pix_loss:
            # Normalize each patch by its own mean and variance
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1e-6).sqrt()

        # MSE loss
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, N) - per-patch MSE

        if self.mae_config.loss_on_masked_only:
            # Average loss over masked patches only
            loss = (loss * mask).sum() / mask.sum()
        else:
            loss = loss.mean()

        # Optional frequency-domain loss for structure preservation
        if self.mae_config.frequency_weight > 0:
            freq_loss = self._frequency_loss(images, pred, mask)
            loss = loss + self.mae_config.frequency_weight * freq_loss

        return loss

    def _frequency_loss(
        self, images: Tensor, pred: Tensor, mask: Tensor
    ) -> Tensor:
        """Compute frequency-domain loss to preserve high-frequency structure.

        Encourages the model to reconstruct fine details (edges, textures)
        that are diagnostically important in medical images.

        Args:
            images: Original images (B, C, H, W).
            pred: Predicted patches (B, N, patch_size^2 * C).
            mask: Binary mask (B, N).

        Returns:
            Scalar frequency loss.
        """
        p = self.mae_config.patch_size
        c = self.mae_config.in_channels
        B, N, _ = pred.shape
        grid_size = int(N**0.5)

        # Reshape predictions back to image format
        pred_img = pred.reshape(B, grid_size, grid_size, p, p, c)
        pred_img = pred_img.permute(0, 5, 1, 3, 2, 4)  # (B, C, h, p, w, p)
        pred_img = pred_img.reshape(B, c, grid_size * p, grid_size * p)

        # Compute 2D FFT
        target_fft = torch.fft.fft2(images, norm="ortho")
        pred_fft = torch.fft.fft2(pred_img, norm="ortho")

        # Spectral loss (L1 on magnitude)
        loss = F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))
        return loss

    @torch.no_grad()
    def extract_features(self, images: Tensor) -> Tensor:
        """Extract features using the full encoder (no masking).

        For downstream evaluation, we pass all patches (no masking)
        through the encoder and take the mean of patch embeddings.

        Args:
            images: Input images (B, C, H, W).

        Returns:
            Feature tensor of shape (B, embed_dim).
        """
        self.backbone.eval()
        patches = self.patch_embed(images) + self.pos_embed
        encoded = self.backbone(patches, is_partial=True)
        if isinstance(encoded, tuple):
            encoded = encoded[0]
        # Global average pooling over patches
        return encoded.mean(dim=1)
