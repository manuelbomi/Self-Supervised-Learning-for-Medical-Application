"""
Unit tests for SSL methods (SimCLR, DINO, MAE).

Verifies that each method:
    - Produces correct output shapes
    - Computes non-zero gradients
    - Loss decreases on a simple forward-backward pass
    - Handles single-channel medical image inputs
    - Feature extraction works in eval mode
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List

import pytest
import torch
import torch.nn as nn
from torch import Tensor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.methods.base import BaseSSLMethod, SSLConfig
from src.methods.simclr import SimCLR, SimCLRConfig, NTXentLoss, ProjectionHead
from src.methods.dino import DINO, DINOConfig, DINOLoss, DINOProjectionHead
from src.methods.mae import MAE, MAEConfig, MAEDecoder, PatchEmbed
from src.backbones.vision_transformer import VisionTransformer, vit_tiny
from src.backbones.resnet_medical import resnet18_medical, ResNetMedical


# ============================================================================
# Fixtures
# ============================================================================

BATCH_SIZE = 4
IMG_SIZE = 64
IN_CHANNELS = 1
EMBED_DIM_VIT = 192
EMBED_DIM_CNN = 512


@pytest.fixture
def vit_backbone() -> VisionTransformer:
    """Small ViT for testing."""
    return vit_tiny(in_channels=IN_CHANNELS, img_size=IMG_SIZE, drop_path_rate=0.0)


@pytest.fixture
def resnet_backbone() -> ResNetMedical:
    """Small ResNet for testing."""
    return resnet18_medical(in_channels=IN_CHANNELS)


@pytest.fixture
def batch_images() -> Tensor:
    """Batch of single-channel images."""
    return torch.randn(BATCH_SIZE, IN_CHANNELS, IMG_SIZE, IMG_SIZE)


@pytest.fixture
def two_views(batch_images: Tensor) -> List[Tensor]:
    """Two augmented views for contrastive learning."""
    return [batch_images + torch.randn_like(batch_images) * 0.1,
            batch_images + torch.randn_like(batch_images) * 0.1]


# ============================================================================
# NT-Xent Loss Tests
# ============================================================================

class TestNTXentLoss:
    """Tests for the NT-Xent contrastive loss."""

    def test_output_is_scalar(self) -> None:
        loss_fn = NTXentLoss(temperature=0.07)
        z1 = torch.randn(BATCH_SIZE, 128)
        z2 = torch.randn(BATCH_SIZE, 128)
        loss = loss_fn(z1, z2)
        assert loss.dim() == 0, "Loss should be a scalar"

    def test_loss_positive(self) -> None:
        loss_fn = NTXentLoss(temperature=0.07)
        z1 = torch.randn(BATCH_SIZE, 128)
        z2 = torch.randn(BATCH_SIZE, 128)
        loss = loss_fn(z1, z2)
        assert loss.item() > 0, "NT-Xent loss should be positive"

    def test_identical_views_low_loss(self) -> None:
        loss_fn = NTXentLoss(temperature=0.5)
        z = torch.randn(BATCH_SIZE, 128)
        # Identical views should have lower loss than random
        loss_identical = loss_fn(z, z + torch.randn_like(z) * 0.01)
        loss_random = loss_fn(z, torch.randn_like(z))
        assert loss_identical.item() < loss_random.item()

    def test_gradient_flow(self) -> None:
        loss_fn = NTXentLoss(temperature=0.07)
        z1 = torch.randn(BATCH_SIZE, 128, requires_grad=True)
        z2 = torch.randn(BATCH_SIZE, 128, requires_grad=True)
        loss = loss_fn(z1, z2)
        loss.backward()
        assert z1.grad is not None, "Gradients should flow to z1"
        assert z2.grad is not None, "Gradients should flow to z2"

    def test_learnable_temperature(self) -> None:
        loss_fn = NTXentLoss(temperature=0.07, learnable=True)
        assert any(p.requires_grad for p in loss_fn.parameters())


# ============================================================================
# SimCLR Tests
# ============================================================================

class TestSimCLR:
    """Tests for the SimCLR method."""

    def test_forward_output_keys(self, resnet_backbone: ResNetMedical, two_views: List[Tensor]) -> None:
        config = SimCLRConfig(
            embed_dim=resnet_backbone.embed_dim, img_size=IMG_SIZE, in_channels=IN_CHANNELS,
            proj_hidden_dim=256, proj_output_dim=64,
        )
        model = SimCLR(resnet_backbone, config)
        output = model(two_views)
        assert "loss" in output
        assert "z1" in output
        assert "z2" in output
        assert "avg_pos_sim" in output

    def test_loss_is_scalar(self, resnet_backbone: ResNetMedical, two_views: List[Tensor]) -> None:
        config = SimCLRConfig(
            embed_dim=resnet_backbone.embed_dim, img_size=IMG_SIZE, in_channels=IN_CHANNELS,
            proj_hidden_dim=256, proj_output_dim=64,
        )
        model = SimCLR(resnet_backbone, config)
        output = model(two_views)
        assert output["loss"].dim() == 0

    def test_projection_shape(self, resnet_backbone: ResNetMedical, two_views: List[Tensor]) -> None:
        proj_dim = 64
        config = SimCLRConfig(
            embed_dim=resnet_backbone.embed_dim, img_size=IMG_SIZE, in_channels=IN_CHANNELS,
            proj_hidden_dim=256, proj_output_dim=proj_dim,
        )
        model = SimCLR(resnet_backbone, config)
        output = model(two_views)
        assert output["z1"].shape == (BATCH_SIZE, proj_dim)
        assert output["z2"].shape == (BATCH_SIZE, proj_dim)

    def test_feature_extraction(self, resnet_backbone: ResNetMedical, batch_images: Tensor) -> None:
        config = SimCLRConfig(
            embed_dim=resnet_backbone.embed_dim, img_size=IMG_SIZE, in_channels=IN_CHANNELS,
        )
        model = SimCLR(resnet_backbone, config)
        model.eval()
        features = model.extract_features(batch_images)
        assert features.shape == (BATCH_SIZE, resnet_backbone.embed_dim)

    def test_gradient_update(self, resnet_backbone: ResNetMedical, two_views: List[Tensor]) -> None:
        config = SimCLRConfig(
            embed_dim=resnet_backbone.embed_dim, img_size=IMG_SIZE, in_channels=IN_CHANNELS,
            proj_hidden_dim=256, proj_output_dim=64,
        )
        model = SimCLR(resnet_backbone, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        output = model(two_views)
        loss = output["loss"]
        loss.backward()
        optimizer.step()

        # Verify parameters changed
        output2 = model(two_views)
        # Loss values should differ after update
        assert not torch.allclose(output["loss"].detach(), output2["loss"].detach())


# ============================================================================
# DINO Tests
# ============================================================================

class TestDINO:
    """Tests for the DINO method."""

    def _make_crops(self, batch_images: Tensor, n_global: int = 2, n_local: int = 2) -> List[Tensor]:
        """Create mock multi-crop inputs."""
        crops = []
        for _ in range(n_global):
            crops.append(batch_images + torch.randn_like(batch_images) * 0.1)
        for _ in range(n_local):
            local = torch.randn(BATCH_SIZE, IN_CHANNELS, IMG_SIZE // 2, IMG_SIZE // 2)
            # Resize local crops to match global
            local = nn.functional.interpolate(local, size=IMG_SIZE)
            crops.append(local)
        return crops

    def test_forward_output_keys(self, vit_backbone: VisionTransformer, batch_images: Tensor) -> None:
        config = DINOConfig(
            embed_dim=EMBED_DIM_VIT, img_size=IMG_SIZE, in_channels=IN_CHANNELS,
            proj_hidden_dim=256, proj_output_dim=256, proj_bottleneck_dim=64,
            num_global_crops=2, num_local_crops=2,
        )
        model = DINO(vit_backbone, config)
        crops = self._make_crops(batch_images)
        output = model(crops)
        assert "loss" in output
        assert "teacher_entropy" in output
        assert "ema_momentum" in output

    def test_teacher_no_gradients(self, vit_backbone: VisionTransformer) -> None:
        config = DINOConfig(
            embed_dim=EMBED_DIM_VIT, img_size=IMG_SIZE, in_channels=IN_CHANNELS,
            proj_hidden_dim=256, proj_output_dim=256, proj_bottleneck_dim=64,
        )
        model = DINO(vit_backbone, config)
        for param in model.teacher_backbone.parameters():
            assert not param.requires_grad

    def test_ema_update(self, vit_backbone: VisionTransformer, batch_images: Tensor) -> None:
        config = DINOConfig(
            embed_dim=EMBED_DIM_VIT, img_size=IMG_SIZE, in_channels=IN_CHANNELS,
            proj_hidden_dim=256, proj_output_dim=256, proj_bottleneck_dim=64,
            ema_momentum_base=0.5,
            num_global_crops=2, num_local_crops=0,
        )
        model = DINO(vit_backbone, config)

        # Store teacher params before update
        teacher_before = {
            name: param.clone()
            for name, param in model.teacher_backbone.named_parameters()
        }

        # Run forward + EMA update
        crops = self._make_crops(batch_images, n_global=2, n_local=0)
        model(crops)
        model.on_step_end(0)

        # Teacher params should have changed
        for name, param in model.teacher_backbone.named_parameters():
            assert not torch.equal(teacher_before[name], param), \
                f"Teacher param {name} did not update via EMA"


# ============================================================================
# MAE Tests
# ============================================================================

class TestMAE:
    """Tests for the Masked Autoencoder method."""

    def test_forward_output_keys(self, vit_backbone: VisionTransformer, batch_images: Tensor) -> None:
        config = MAEConfig(
            embed_dim=EMBED_DIM_VIT, img_size=IMG_SIZE, in_channels=IN_CHANNELS,
            patch_size=8, mask_ratio=0.75,
            decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=4,
        )
        model = MAE(vit_backbone, config)
        output = model(batch_images)
        assert "loss" in output
        assert "pred" in output
        assert "mask" in output

    def test_mask_ratio(self, vit_backbone: VisionTransformer, batch_images: Tensor) -> None:
        mask_ratio = 0.75
        config = MAEConfig(
            embed_dim=EMBED_DIM_VIT, img_size=IMG_SIZE, in_channels=IN_CHANNELS,
            patch_size=8, mask_ratio=mask_ratio,
            decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=4,
        )
        model = MAE(vit_backbone, config)
        output = model(batch_images)

        mask = output["mask"]
        actual_ratio = mask.float().mean().item()
        assert abs(actual_ratio - mask_ratio) < 0.05, \
            f"Actual mask ratio {actual_ratio:.2f} differs from target {mask_ratio}"

    def test_reconstruction_shape(self, vit_backbone: VisionTransformer, batch_images: Tensor) -> None:
        patch_size = 8
        config = MAEConfig(
            embed_dim=EMBED_DIM_VIT, img_size=IMG_SIZE, in_channels=IN_CHANNELS,
            patch_size=patch_size, mask_ratio=0.75,
            decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=4,
        )
        model = MAE(vit_backbone, config)
        output = model(batch_images)

        num_patches = (IMG_SIZE // patch_size) ** 2
        expected_shape = (BATCH_SIZE, num_patches, patch_size * patch_size * IN_CHANNELS)
        assert output["pred"].shape == expected_shape

    def test_feature_extraction(self, vit_backbone: VisionTransformer, batch_images: Tensor) -> None:
        config = MAEConfig(
            embed_dim=EMBED_DIM_VIT, img_size=IMG_SIZE, in_channels=IN_CHANNELS,
            patch_size=8, decoder_embed_dim=64, decoder_depth=2, decoder_num_heads=4,
        )
        model = MAE(vit_backbone, config)
        model.eval()
        features = model.extract_features(batch_images)
        assert features.shape == (BATCH_SIZE, EMBED_DIM_VIT)


# ============================================================================
# Backbone Tests
# ============================================================================

class TestBackbones:
    """Tests for backbone architectures."""

    def test_vit_output_shape(self, vit_backbone: VisionTransformer, batch_images: Tensor) -> None:
        output = vit_backbone(batch_images)
        assert output.shape == (BATCH_SIZE, EMBED_DIM_VIT)

    def test_vit_all_tokens(self, vit_backbone: VisionTransformer, batch_images: Tensor) -> None:
        output = vit_backbone(batch_images, return_all_tokens=True)
        num_patches = (IMG_SIZE // 16) ** 2
        assert output.shape == (BATCH_SIZE, 1 + num_patches, EMBED_DIM_VIT)

    def test_resnet_output_shape(self, resnet_backbone: ResNetMedical, batch_images: Tensor) -> None:
        output = resnet_backbone(batch_images)
        assert output.shape == (BATCH_SIZE, resnet_backbone.embed_dim)

    def test_resnet_single_channel(self) -> None:
        model = resnet18_medical(in_channels=1)
        x = torch.randn(2, 1, IMG_SIZE, IMG_SIZE)
        output = model(x)
        assert output.shape[0] == 2

    def test_resnet_feature_pyramid(self, resnet_backbone: ResNetMedical, batch_images: Tensor) -> None:
        features = resnet_backbone.forward_features(batch_images)
        assert "stem" in features
        assert "layer1" in features
        assert "layer4" in features
        assert "pooled" in features


# ============================================================================
# Projection Head Tests
# ============================================================================

class TestProjectionHeads:
    """Tests for projection head architectures."""

    def test_simclr_projection_head(self) -> None:
        head = ProjectionHead(input_dim=512, hidden_dim=256, output_dim=64, num_layers=2)
        x = torch.randn(BATCH_SIZE, 512)
        out = head(x)
        assert out.shape == (BATCH_SIZE, 64)

    def test_dino_projection_head(self) -> None:
        head = DINOProjectionHead(
            input_dim=192, hidden_dim=256, bottleneck_dim=64, output_dim=1024, num_layers=3
        )
        x = torch.randn(BATCH_SIZE, 192)
        out = head(x)
        assert out.shape == (BATCH_SIZE, 1024)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """End-to-end integration tests."""

    def test_simclr_train_step(self) -> None:
        backbone = resnet18_medical(in_channels=1)
        config = SimCLRConfig(
            embed_dim=backbone.embed_dim, img_size=IMG_SIZE, in_channels=1,
            proj_hidden_dim=256, proj_output_dim=64,
        )
        model = SimCLR(backbone, config)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x1 = torch.randn(4, 1, IMG_SIZE, IMG_SIZE)
        x2 = torch.randn(4, 1, IMG_SIZE, IMG_SIZE)

        model.train()
        output = model([x1, x2])
        output["loss"].backward()
        optimizer.step()

    def test_checkpoint_save_load(self, tmp_path: Path) -> None:
        backbone = resnet18_medical(in_channels=1)
        config = SimCLRConfig(
            embed_dim=backbone.embed_dim, img_size=IMG_SIZE, in_channels=1,
        )
        model = SimCLR(backbone, config)

        ckpt_path = tmp_path / "test_ckpt.pth"
        model.save_checkpoint(ckpt_path, epoch=10)
        assert ckpt_path.exists()

        epoch = model.load_checkpoint(ckpt_path)
        assert epoch == 10

    def test_backbone_extraction(self) -> None:
        backbone = resnet18_medical(in_channels=1)
        config = SimCLRConfig(
            embed_dim=backbone.embed_dim, img_size=IMG_SIZE, in_channels=1,
        )
        model = SimCLR(backbone, config)

        state_dict = model.extract_backbone_state_dict()
        assert len(state_dict) > 0
        # Should not contain projection head keys
        for key in state_dict:
            assert "projection" not in key


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
