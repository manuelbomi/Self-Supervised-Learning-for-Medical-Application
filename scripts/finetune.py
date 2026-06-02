#!/usr/bin/env python3
"""
Fine-tune a pretrained SSL encoder on downstream medical imaging tasks.

Loads a self-supervised pretrained backbone and fine-tunes it (with an
optional linear warmup of the classifier head) on labeled data for
classification, detection, or segmentation tasks.

Supports:
    - Full fine-tuning (all layers trainable)
    - Partial fine-tuning (freeze early layers)
    - Linear evaluation (freeze all backbone layers)
    - Label-efficient fine-tuning with configurable label fractions

Usage:
    python scripts/finetune.py \
        --checkpoint checkpoints/dino_ep200.pth \
        --method dino \
        --task classification \
        --label-fraction 0.1 \
        --data-dir data/mammography \
        --label-file data/mammography/labels.csv \
        --epochs 50
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.methods.base import BaseSSLMethod
from src.augmentations.medical_augmentations import (
    MedicalAugConfig,
    build_simclr_augmentation,
    build_eval_augmentation,
)
from src.data.medical_dataset import MedicalImageDataset, DatasetConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class FineTuneClassifier(nn.Module):
    """Classification head for fine-tuning pretrained SSL backbone.

    Supports different fine-tuning strategies:
        - 'full': all backbone layers are trainable
        - 'partial': freeze early layers, tune later layers
        - 'linear': freeze all backbone layers (linear evaluation)

    Args:
        backbone: Pretrained backbone network.
        embed_dim: Backbone output dimensionality.
        num_classes: Number of target classes.
        strategy: Fine-tuning strategy ('full', 'partial', 'linear').
        freeze_layers: Number of early layers to freeze (for 'partial').
        dropout: Dropout rate before the classifier head.
    """

    def __init__(
        self,
        backbone: nn.Module,
        embed_dim: int,
        num_classes: int,
        strategy: str = "full",
        freeze_layers: int = 0,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.strategy = strategy

        # Classification head
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

        # Apply freezing strategy
        self._apply_strategy(strategy, freeze_layers)

        # Initialize head
        nn.init.normal_(self.head[-1].weight, 0, 0.01)
        nn.init.zeros_(self.head[-1].bias)

    def _apply_strategy(self, strategy: str, freeze_layers: int) -> None:
        """Apply the fine-tuning strategy to the backbone."""
        if strategy == "linear":
            for param in self.backbone.parameters():
                param.requires_grad = False
            logger.info("Linear evaluation: all backbone layers frozen")

        elif strategy == "partial":
            # Freeze first N layers
            frozen_count = 0
            for name, param in self.backbone.named_parameters():
                layer_idx = self._get_layer_index(name)
                if layer_idx is not None and layer_idx < freeze_layers:
                    param.requires_grad = False
                    frozen_count += 1

            total = sum(1 for _ in self.backbone.parameters())
            logger.info(
                f"Partial fine-tuning: froze {frozen_count}/{total} parameters "
                f"(first {freeze_layers} layers)"
            )

        elif strategy == "full":
            logger.info("Full fine-tuning: all parameters trainable")
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    @staticmethod
    def _get_layer_index(param_name: str) -> Optional[int]:
        """Extract layer index from parameter name.

        Handles both ViT (blocks.N) and ResNet (layerN) naming conventions.
        """
        import re
        match = re.search(r"(?:blocks|layer)\.?(\d+)", param_name)
        if match:
            return int(match.group(1))
        return None

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: backbone features -> classification logits.

        Args:
            x: Input images (B, C, H, W).

        Returns:
            Class logits (B, num_classes).
        """
        features = self.backbone(x)
        # Handle ViT CLS token
        if features.dim() == 3:
            features = features[:, 0]
        return self.head(features)


class FineTuner:
    """Fine-tuning pipeline for pretrained SSL models.

    Args:
        model: FineTuneClassifier instance.
        device: Compute device.
        learning_rate: Base learning rate.
        weight_decay: L2 regularization.
        epochs: Number of fine-tuning epochs.
        warmup_epochs: Number of warmup epochs (head-only training).
        use_class_weights: Whether to use class-balanced loss.
    """

    def __init__(
        self,
        model: FineTuneClassifier,
        device: torch.device,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        epochs: int = 50,
        warmup_epochs: int = 5,
        use_class_weights: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.use_class_weights = use_class_weights

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer with layer-wise learning rate decay."""
        head_params = list(self.model.head.parameters())
        backbone_params = [
            p for p in self.model.backbone.parameters() if p.requires_grad
        ]

        param_groups = [
            {"params": head_params, "lr": self.learning_rate * 10},
            {"params": backbone_params, "lr": self.learning_rate},
        ]

        return torch.optim.AdamW(
            param_groups, weight_decay=self.weight_decay
        )

    def _compute_class_weights(
        self, dataloader: DataLoader
    ) -> Optional[Tensor]:
        """Compute inverse-frequency class weights for balanced loss."""
        if not self.use_class_weights:
            return None

        label_counts: Dict[int, int] = {}
        for _, labels in dataloader:
            for label in labels.tolist():
                if label >= 0:
                    label_counts[label] = label_counts.get(label, 0) + 1

        if not label_counts:
            return None

        total = sum(label_counts.values())
        num_classes = max(label_counts.keys()) + 1
        weights = torch.zeros(num_classes)
        for cls, count in label_counts.items():
            weights[cls] = total / (num_classes * count)

        logger.info(f"Class weights: {weights.tolist()}")
        return weights.to(self.device)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """Run the fine-tuning loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.

        Returns:
            Training history and best metrics.
        """
        optimizer = self._build_optimizer()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.epochs
        )

        class_weights = self._compute_class_weights(train_loader)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        best_metric = 0.0
        best_epoch = 0
        history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [], "val_acc": [], "val_auc": [],
        }

        for epoch in range(self.epochs):
            # Warmup: train only head for first N epochs
            if epoch < self.warmup_epochs and self.model.strategy != "linear":
                for param in self.model.backbone.parameters():
                    param.requires_grad = False
            elif epoch == self.warmup_epochs and self.model.strategy == "full":
                for param in self.model.backbone.parameters():
                    param.requires_grad = True
                logger.info(f"Epoch {epoch + 1}: unfreezing backbone")

            # Training
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                logits = self.model(images)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += images.size(0)

            scheduler.step()

            train_loss = total_loss / max(total, 1)
            train_acc = correct / max(total, 1)
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # Validation
            val_metrics = self._evaluate(val_loader, criterion)
            history["val_acc"].append(val_metrics["accuracy"])
            history["val_auc"].append(val_metrics.get("auc", 0.0))

            # Track best
            metric = val_metrics.get("auc", val_metrics["accuracy"])
            if metric > best_metric:
                best_metric = metric
                best_epoch = epoch

            logger.info(
                f"Epoch {epoch + 1}/{self.epochs} | "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Val AUC: {val_metrics.get('auc', 0.0):.4f} | "
                f"Best: {best_metric:.4f} (ep {best_epoch + 1})"
            )

        return {
            "best_metric": best_metric,
            "best_epoch": best_epoch,
            "history": history,
        }

    @torch.no_grad()
    def _evaluate(
        self, dataloader: DataLoader, criterion: nn.Module
    ) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        all_logits, all_labels = [], []

        for images, labels in dataloader:
            images = images.to(self.device)
            logits = self.model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)

        logits = torch.cat(all_logits, dim=0)
        labels = torch.cat(all_labels, dim=0)

        preds = logits.argmax(dim=1)
        accuracy = (preds == labels).float().mean().item()

        metrics: Dict[str, float] = {"accuracy": accuracy}

        # AUC for binary classification
        num_classes = logits.shape[1]
        if num_classes == 2:
            try:
                from sklearn.metrics import roc_auc_score
                probs = F.softmax(logits, dim=1)[:, 1].numpy()
                metrics["auc"] = roc_auc_score(labels.numpy(), probs)
            except (ImportError, ValueError):
                pass

        return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune SSL model")

    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--method", type=str, required=True,
                        choices=["simclr", "dino", "mae"])
    parser.add_argument("--task", type=str, default="classification",
                        choices=["classification"])

    parser.add_argument("--data-dir", type=str, default="data/mammography")
    parser.add_argument("--label-file", type=str, default=None)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=2)

    parser.add_argument("--strategy", type=str, default="full",
                        choices=["full", "partial", "linear"])
    parser.add_argument("--freeze-layers", type=int, default=6)
    parser.add_argument("--label-fraction", type=float, default=1.0)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--weight-decay", type=float, default=0.01)

    parser.add_argument("--output-dir", type=str, default="results")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained backbone
    from scripts.evaluate_representations import load_ssl_model
    ssl_model = load_ssl_model(args.checkpoint, args.method, device)

    # Extract backbone
    if hasattr(ssl_model, "backbone"):
        backbone = ssl_model.backbone
    elif hasattr(ssl_model, "module"):
        backbone = ssl_model.module.backbone
    else:
        raise ValueError("Cannot extract backbone from model")

    embed_dim = getattr(backbone, "embed_dim", 384)

    # Build fine-tuning model
    ft_model = FineTuneClassifier(
        backbone=backbone,
        embed_dim=embed_dim,
        num_classes=args.num_classes,
        strategy=args.strategy,
        freeze_layers=args.freeze_layers,
    )

    # Build data transforms
    aug_config = MedicalAugConfig(img_size=args.img_size)
    train_transform = build_simclr_augmentation(aug_config)
    val_transform = build_eval_augmentation(aug_config)

    # Build datasets
    train_config = DatasetConfig(
        data_dir=args.data_dir,
        img_size=args.img_size,
        label_file=args.label_file,
        label_fraction=args.label_fraction,
    )
    val_config = DatasetConfig(
        data_dir=args.data_dir,
        img_size=args.img_size,
        label_file=args.label_file,
        split="val",
    )

    train_dataset = MedicalImageDataset(train_config, transform=train_transform)
    val_dataset = MedicalImageDataset(val_config, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True,
    )

    # Fine-tune
    finetuner = FineTuner(
        model=ft_model,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
    )

    results = finetuner.train(train_loader, val_loader)

    logger.info(f"\nFine-tuning complete!")
    logger.info(f"Best metric: {results['best_metric']:.4f} (epoch {results['best_epoch'] + 1})")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(results, output_dir / f"finetune_{args.method}_{args.strategy}.pt")


if __name__ == "__main__":
    main()
