"""
Vision Transformer (ViT) backbone for medical image self-supervised learning.

Implements a standard ViT with adaptations for medical imaging:
    - Single-channel (grayscale) patch embedding
    - Flexible position embedding interpolation for varying resolutions
    - Support for partial patch input (used by MAE)
    - CLS token for global representation

Reference:
    Dosovitskiy, A. et al. "An Image is Worth 16x16 Words." ICLR 2021.
"""

from __future__ import annotations

import logging
import math
from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)


class PatchEmbedding(nn.Module):
    """Project image patches to embedding vectors.

    Splits the input image into non-overlapping patches and projects each
    patch to a D-dimensional embedding using a single convolution.

    Args:
        img_size: Input image spatial size (assumed square).
        patch_size: Size of each square patch.
        in_channels: Number of input channels (1 for medical grayscale).
        embed_dim: Output embedding dimensionality.
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
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, (
            f"Input size ({H}x{W}) doesn't match expected ({self.img_size}x{self.img_size})"
        )
        x = self.proj(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x


class Attention(nn.Module):
    """Multi-head self-attention with optional attention map output.

    Args:
        dim: Input/output dimensionality.
        num_heads: Number of attention heads.
        qkv_bias: Whether to include bias in QKV projection.
        attn_drop: Dropout rate on attention weights.
        proj_drop: Dropout rate on output projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(
        self, x: Tensor, return_attention: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Compute multi-head self-attention.

        Args:
            x: Input tensor of shape (B, N, D).
            return_attention: If True, also return attention weights.

        Returns:
            Output tensor of shape (B, N, D), optionally with attention
            weights of shape (B, num_heads, N, N).
        """
        B, N, D = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # Each: (B, heads, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x


class FeedForward(nn.Module):
    """Feed-forward network (MLP) with GELU activation.

    Args:
        dim: Input/output dimensionality.
        hidden_dim: Hidden layer width (typically 4 * dim).
        drop: Dropout rate.
    """

    def __init__(
        self, dim: int, hidden_dim: Optional[int] = None, drop: float = 0.0
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block with pre-norm architecture.

    Uses pre-LayerNorm (norm before attention/FFN) which is more stable
    for training compared to post-norm.

    Args:
        dim: Embedding dimensionality.
        num_heads: Number of attention heads.
        mlp_ratio: Ratio of FFN hidden dim to embedding dim.
        qkv_bias: Whether QKV projection has bias.
        drop: Dropout rate.
        attn_drop: Attention dropout rate.
        drop_path: Stochastic depth rate.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        drop_path: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias,
            attn_drop=attn_drop, proj_drop=drop,
        )
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = FeedForward(dim, hidden_dim=int(dim * mlp_ratio), drop=drop)

        # Stochastic depth
        self.drop_path_rate = drop_path
        if drop_path > 0.0:
            self.drop_path_fn = self._stochastic_depth
        else:
            self.drop_path_fn = nn.Identity()

    def _stochastic_depth(self, x: Tensor) -> Tensor:
        """Apply stochastic depth (drop entire residual branch)."""
        if not self.training:
            return x
        keep_prob = 1 - self.drop_path_rate
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep_prob, device=x.device))
        return x * mask / keep_prob

    def forward(
        self, x: Tensor, return_attention: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        """Forward pass through the transformer block.

        Args:
            x: Input tensor of shape (B, N, D).
            return_attention: If True, return attention weights from this block.

        Returns:
            Output tensor (B, N, D), optionally with attention weights.
        """
        if return_attention:
            attn_out, attn_weights = self.attn(self.norm1(x), return_attention=True)
            x = x + self.drop_path_fn(attn_out)
            x = x + self.drop_path_fn(self.mlp(self.norm2(x)))
            return x, attn_weights

        x = x + self.drop_path_fn(self.attn(self.norm1(x)))
        x = x + self.drop_path_fn(self.mlp(self.norm2(x)))
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer backbone for medical image SSL.

    A standard ViT architecture with medical imaging adaptations:
        - Single-channel patch embedding (grayscale medical images)
        - Position embedding interpolation for resolution flexibility
        - Support for partial patch sequences (MAE encoder mode)
        - CLS token for global image representation

    Args:
        img_size: Input image size (square).
        patch_size: Patch size for tokenization.
        in_channels: Number of input channels (1 for grayscale).
        embed_dim: Transformer embedding dimension.
        depth: Number of transformer blocks.
        num_heads: Number of attention heads.
        mlp_ratio: FFN hidden dim ratio.
        qkv_bias: QKV bias in attention.
        drop_rate: Dropout rate.
        attn_drop_rate: Attention dropout rate.
        drop_path_rate: Stochastic depth rate.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 1,
        embed_dim: int = 384,
        depth: int = 12,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
        )

        # CLS token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 1 + self.num_patches, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
            )
            for i in range(depth)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights following ViT conventions."""
        # Position embedding: truncated normal
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Linear layers and LayerNorm
        self.apply(self._init_module)

    @staticmethod
    def _init_module(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def interpolate_pos_embed(self, x: Tensor, h: int, w: int) -> Tensor:
        """Interpolate position embeddings for different input resolutions.

        Enables using a model trained at one resolution for inference at
        a different resolution -- critical for medical imaging where study
        sizes vary (e.g., different mammogram dimensions).

        Args:
            x: Current patch embeddings (B, N, D).
            h: Number of patches in height.
            w: Number of patches in width.

        Returns:
            Interpolated position embeddings (1, 1 + h*w, D).
        """
        num_patches = h * w
        if num_patches == self.num_patches:
            return self.pos_embed

        cls_pos = self.pos_embed[:, :1]
        patch_pos = self.pos_embed[:, 1:]

        dim = patch_pos.shape[-1]
        orig_size = int(self.num_patches**0.5)

        # Reshape to 2D grid, interpolate, flatten back
        patch_pos = patch_pos.reshape(1, orig_size, orig_size, dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(
            patch_pos, size=(h, w), mode="bicubic", align_corners=False
        )
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, -1, dim)

        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(
        self,
        x: Tensor,
        is_partial: bool = False,
        return_all_tokens: bool = False,
    ) -> Tensor:
        """Forward pass through the Vision Transformer.

        Args:
            x: Input tensor. If is_partial=False: images (B, C, H, W).
                If is_partial=True: pre-embedded patches (B, N_vis, D).
            is_partial: If True, input is already embedded (for MAE encoder).
            return_all_tokens: If True, return all token embeddings including
                patch tokens. Otherwise, return only CLS token.

        Returns:
            If return_all_tokens: (B, 1 + N, D) -- all token embeddings.
            Otherwise: (B, D) -- CLS token embedding.
        """
        if is_partial:
            # Input is already embedded patches (MAE mode)
            # No CLS token, no positional embedding (handled by MAE)
            for block in self.blocks:
                x = block(x)
            x = self.norm(x)
            return x

        # Standard forward: image -> patches -> transformer
        x = self.patch_embed(x)  # (B, N, D)

        # Prepend CLS token
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, 1+N, D)

        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        if return_all_tokens:
            return x  # (B, 1+N, D)
        return x[:, 0]  # CLS token: (B, D)

    def get_attention_maps(self, x: Tensor, layer: int = -1) -> Tensor:
        """Extract attention maps from a specific layer.

        Useful for visualizing what the model attends to in medical images
        (e.g., does it focus on lesion regions?).

        Args:
            x: Input images (B, C, H, W).
            layer: Which transformer block to extract from (-1 = last).

        Returns:
            Attention weights of shape (B, num_heads, N+1, N+1).
        """
        x = self.patch_embed(x)
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        target_layer = layer if layer >= 0 else len(self.blocks) + layer

        for i, block in enumerate(self.blocks):
            if i == target_layer:
                _, attn = block(x, return_attention=True)
                return attn
            x = block(x)

        raise ValueError(f"Layer {layer} not found")


def vit_tiny(in_channels: int = 1, img_size: int = 224, **kwargs) -> VisionTransformer:
    """ViT-Tiny: 5.7M parameters."""
    return VisionTransformer(
        img_size=img_size, in_channels=in_channels,
        embed_dim=192, depth=12, num_heads=3, **kwargs,
    )


def vit_small(in_channels: int = 1, img_size: int = 224, **kwargs) -> VisionTransformer:
    """ViT-Small: 22M parameters. Good balance for medical imaging."""
    return VisionTransformer(
        img_size=img_size, in_channels=in_channels,
        embed_dim=384, depth=12, num_heads=6, **kwargs,
    )


def vit_base(in_channels: int = 1, img_size: int = 224, **kwargs) -> VisionTransformer:
    """ViT-Base: 86M parameters."""
    return VisionTransformer(
        img_size=img_size, in_channels=in_channels,
        embed_dim=768, depth=12, num_heads=12, **kwargs,
    )
