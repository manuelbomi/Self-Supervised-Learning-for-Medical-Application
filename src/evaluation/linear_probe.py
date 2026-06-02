"""
Linear evaluation protocol for self-supervised representations.

Trains a single linear layer on top of frozen SSL features to evaluate
representation quality. This is the standard protocol for benchmarking
SSL methods: better representations yield higher linear probe accuracy.

The key insight is that if a linear classifier achieves high accuracy,
the SSL features must encode class-relevant information in a linearly
separable manner.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class LinearProbeConfig:
    """Configuration for linear evaluation."""

    # Architecture
    embed_dim: int = 384
    num_classes: int = 2
    use_bn: bool = False  # BN before linear can help with feature scale

    # Optimizer
    lr: float = 0.1
    weight_decay: float = 0.0
    momentum: float = 0.9
    optimizer: str = "sgd"

    # Training
    epochs: int = 100
    batch_size: int = 256
    warmup_epochs: int = 5

    # Scheduler
    scheduler: str = "cosine"  # "cosine" or "step"
    step_lr_milestones: List[int] = None
    step_lr_gamma: float = 0.1

    # Evaluation
    eval_every: int = 5
    early_stop_patience: int = 20


class LinearClassifier(nn.Module):
    """Single linear layer classifier for evaluating frozen representations.

    Optionally applies BatchNorm before the linear layer to handle
    scale differences between SSL methods.

    Args:
        embed_dim: Dimensionality of input features.
        num_classes: Number of target classes.
        use_bn: Whether to apply BN before the linear layer.
    """

    def __init__(
        self, embed_dim: int, num_classes: int, use_bn: bool = False
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        if use_bn:
            layers.append(nn.BatchNorm1d(embed_dim, affine=False))
        layers.append(nn.Linear(embed_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

        # Initialize linear layer
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, features: Tensor) -> Tensor:
        """Classify input features.

        Args:
            features: Frozen backbone features (B, embed_dim).

        Returns:
            Class logits (B, num_classes).
        """
        return self.classifier(features)


class LinearProbeEvaluator:
    """Evaluates SSL representations via linear probe protocol.

    Freezes the pretrained backbone, extracts features for the entire
    dataset, then trains a linear classifier on those features.

    Args:
        ssl_model: Pretrained SSL model (will be frozen).
        config: Linear probe configuration.
        device: Device to run evaluation on.
    """

    def __init__(
        self,
        ssl_model: nn.Module,
        config: LinearProbeConfig,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.ssl_model = ssl_model
        self.config = config
        self.device = device

        # Freeze SSL model
        self.ssl_model.eval()
        for param in self.ssl_model.parameters():
            param.requires_grad = False

        # Linear classifier
        self.classifier = LinearClassifier(
            embed_dim=config.embed_dim,
            num_classes=config.num_classes,
            use_bn=config.use_bn,
        ).to(device)

    @torch.no_grad()
    def extract_features(
        self, dataloader: DataLoader
    ) -> Tuple[Tensor, Tensor]:
        """Extract features from the frozen backbone for the entire dataset.

        Args:
            dataloader: DataLoader yielding (images, labels).

        Returns:
            Tuple of (all_features, all_labels).
        """
        self.ssl_model.eval()
        all_features: List[Tensor] = []
        all_labels: List[Tensor] = []

        for images, labels in dataloader:
            images = images.to(self.device)
            features = self.ssl_model.extract_features(images)
            all_features.append(features.cpu())
            all_labels.append(labels)

        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for the linear classifier."""
        if self.config.optimizer == "sgd":
            return torch.optim.SGD(
                self.classifier.parameters(),
                lr=self.config.lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer == "adam":
            return torch.optim.Adam(
                self.classifier.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _build_scheduler(
        self, optimizer: torch.optim.Optimizer
    ) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler."""
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.epochs
            )
        elif self.config.scheduler == "step":
            milestones = self.config.step_lr_milestones or [60, 80]
            return torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=self.config.step_lr_gamma
            )
        return None

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, Any]:
        """Train linear probe and evaluate on validation set.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.

        Returns:
            Dictionary with training history and best metrics.
        """
        logger.info("Extracting features for linear probe evaluation...")

        # Pre-extract all features (much faster than re-encoding each epoch)
        train_features, train_labels = self.extract_features(train_loader)
        val_features, val_labels = self.extract_features(val_loader)

        train_features = train_features.to(self.device)
        train_labels = train_labels.to(self.device)
        val_features = val_features.to(self.device)
        val_labels = val_labels.to(self.device)

        # Create feature-only data loader
        train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
        feature_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True
        )

        optimizer = self._build_optimizer()
        scheduler = self._build_scheduler(optimizer)
        criterion = nn.CrossEntropyLoss()

        best_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [], "val_acc": [], "val_auc": [],
        }

        for epoch in range(self.config.epochs):
            # Training
            self.classifier.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for feats, labels in feature_loader:
                logits = self.classifier(feats)
                loss = criterion(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * feats.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += feats.size(0)

            if scheduler is not None:
                scheduler.step()

            train_loss = total_loss / total
            train_acc = correct / total

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)

            # Validation
            if (epoch + 1) % self.config.eval_every == 0 or epoch == self.config.epochs - 1:
                val_metrics = self._evaluate(val_features, val_labels)
                val_acc = val_metrics["accuracy"]
                history["val_acc"].append(val_acc)
                history["val_auc"].append(val_metrics.get("auc", 0.0))

                if val_acc > best_acc:
                    best_acc = val_acc
                    best_epoch = epoch
                    patience_counter = 0
                else:
                    patience_counter += 1

                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} | "
                    f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
                    f"Val Acc: {val_acc:.4f} | Best: {best_acc:.4f} (ep {best_epoch + 1})"
                )

                if patience_counter >= self.config.early_stop_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        return {
            "best_accuracy": best_acc,
            "best_epoch": best_epoch,
            "history": history,
        }

    @torch.no_grad()
    def _evaluate(self, features: Tensor, labels: Tensor) -> Dict[str, float]:
        """Evaluate linear probe on a set of features.

        Args:
            features: Feature tensor (N, embed_dim).
            labels: Label tensor (N,).

        Returns:
            Dictionary with accuracy and optionally AUC.
        """
        self.classifier.eval()
        logits = self.classifier(features)
        probs = F.softmax(logits, dim=1)

        preds = logits.argmax(dim=1)
        accuracy = (preds == labels).float().mean().item()

        metrics: Dict[str, float] = {"accuracy": accuracy}

        # Compute AUC for binary classification
        if self.config.num_classes == 2:
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(
                    labels.cpu().numpy(), probs[:, 1].cpu().numpy()
                )
                metrics["auc"] = auc
            except (ImportError, ValueError):
                pass

        return metrics
