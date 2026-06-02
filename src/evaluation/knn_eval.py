"""
k-Nearest Neighbors evaluation of SSL embedding quality.

Provides a non-parametric evaluation of learned representations using
weighted k-NN classification. Unlike linear probe, k-NN requires no
training and directly measures the local structure of the embedding space.

A good SSL representation should place semantically similar images close
together in embedding space, yielding high k-NN accuracy.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class KNNConfig:
    """Configuration for k-NN evaluation."""

    k_values: List[int] = field(default_factory=lambda: [1, 5, 10, 20, 50, 100, 200])
    temperature: float = 0.07
    num_classes: int = 2
    batch_size: int = 256
    use_cosine: bool = True  # Cosine vs Euclidean distance


class KNNEvaluator:
    """Evaluate SSL representations using weighted k-NN classification.

    Uses the training set features as a memory bank. For each test sample,
    finds the k nearest neighbors in the training features and predicts
    the class using distance-weighted voting.

    The temperature parameter controls the sharpness of the distance
    weighting: lower temperature gives more weight to the closest neighbors.

    Args:
        ssl_model: Pretrained SSL model (will be frozen for feature extraction).
        config: k-NN evaluation configuration.
        device: Compute device.
    """

    def __init__(
        self,
        ssl_model: nn.Module,
        config: KNNConfig,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        self.ssl_model = ssl_model
        self.config = config
        self.device = device

        # Freeze SSL model
        self.ssl_model.eval()
        for param in self.ssl_model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def extract_features(
        self, dataloader: DataLoader
    ) -> Tuple[Tensor, Tensor]:
        """Extract normalized features from the frozen backbone.

        Args:
            dataloader: DataLoader yielding (images, labels).

        Returns:
            Tuple of (features, labels), features are L2-normalized.
        """
        self.ssl_model.eval()
        all_features: List[Tensor] = []
        all_labels: List[Tensor] = []

        for images, labels in dataloader:
            images = images.to(self.device)
            features = self.ssl_model.extract_features(images)

            if self.config.use_cosine:
                features = F.normalize(features, dim=1)

            all_features.append(features.cpu())
            all_labels.append(labels)

        features = torch.cat(all_features, dim=0)
        labels = torch.cat(all_labels, dim=0)

        logger.info(
            f"Extracted features: {features.shape} from {len(dataloader.dataset)} samples"
        )
        return features, labels

    @torch.no_grad()
    def _knn_predict(
        self,
        test_features: Tensor,
        train_features: Tensor,
        train_labels: Tensor,
        k: int,
    ) -> Tuple[Tensor, Tensor]:
        """Predict labels for test samples using weighted k-NN.

        For each test sample, computes similarity to all training samples,
        selects the top-k neighbors, and performs temperature-scaled
        distance-weighted voting.

        Args:
            test_features: Test feature matrix (N_test, D).
            train_features: Training feature matrix (N_train, D).
            train_labels: Training labels (N_train,).
            k: Number of neighbors.

        Returns:
            Tuple of (predicted_labels, class_probabilities).
        """
        num_test = test_features.shape[0]
        num_classes = self.config.num_classes
        temperature = self.config.temperature

        all_probs = []
        chunk_size = 128  # Process in chunks to avoid OOM

        for start in range(0, num_test, chunk_size):
            end = min(start + chunk_size, num_test)
            chunk = test_features[start:end].to(self.device)
            train_feat = train_features.to(self.device)

            # Compute similarity matrix
            if self.config.use_cosine:
                # Cosine similarity (features are already normalized)
                sim = torch.mm(chunk, train_feat.t())
            else:
                # Negative L2 distance
                sim = -torch.cdist(chunk, train_feat, p=2)

            # Get top-k neighbors
            top_k_sim, top_k_idx = sim.topk(k, dim=1, largest=True)  # (chunk, k)

            # Temperature-scaled weights
            weights = (top_k_sim / temperature).exp()  # (chunk, k)

            # Get neighbor labels
            neighbor_labels = train_labels.to(self.device)[top_k_idx]  # (chunk, k)

            # Weighted voting: accumulate weights per class
            probs = torch.zeros(chunk.size(0), num_classes, device=self.device)
            for c in range(num_classes):
                mask = (neighbor_labels == c).float()  # (chunk, k)
                probs[:, c] = (weights * mask).sum(dim=1)

            # Normalize to probabilities
            probs = probs / probs.sum(dim=1, keepdim=True).clamp(min=1e-8)
            all_probs.append(probs.cpu())

        all_probs = torch.cat(all_probs, dim=0)  # (N_test, num_classes)
        predictions = all_probs.argmax(dim=1)

        return predictions, all_probs

    def evaluate(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
    ) -> Dict[str, Any]:
        """Run k-NN evaluation for all configured k values.

        Args:
            train_loader: Training data loader (memory bank).
            test_loader: Test data loader (query samples).

        Returns:
            Dictionary with:
                - results: {k: accuracy} for each k value
                - best_k: k value with highest accuracy
                - best_accuracy: highest accuracy achieved
                - class_probabilities: probabilities for best k
        """
        logger.info("Extracting training features for k-NN memory bank...")
        train_features, train_labels = self.extract_features(train_loader)

        logger.info("Extracting test features...")
        test_features, test_labels = self.extract_features(test_loader)

        results: Dict[int, float] = {}
        best_acc = 0.0
        best_k = 0
        best_probs = None

        for k in self.config.k_values:
            if k > len(train_features):
                logger.warning(
                    f"Skipping k={k}: exceeds training set size ({len(train_features)})"
                )
                continue

            predictions, probs = self._knn_predict(
                test_features, train_features, train_labels, k
            )

            accuracy = (predictions == test_labels).float().mean().item()
            results[k] = accuracy

            if accuracy > best_acc:
                best_acc = accuracy
                best_k = k
                best_probs = probs

            logger.info(f"k-NN (k={k:>3d}): accuracy = {accuracy:.4f}")

        # Compute AUC for binary classification with best k
        auc = None
        if self.config.num_classes == 2 and best_probs is not None:
            try:
                from sklearn.metrics import roc_auc_score
                auc = roc_auc_score(
                    test_labels.numpy(), best_probs[:, 1].numpy()
                )
                logger.info(f"k-NN (k={best_k}) AUC = {auc:.4f}")
            except (ImportError, ValueError):
                pass

        return {
            "results": results,
            "best_k": best_k,
            "best_accuracy": best_acc,
            "auc": auc,
            "class_probabilities": best_probs,
        }

    def compute_retrieval_metrics(
        self,
        query_features: Tensor,
        query_labels: Tensor,
        gallery_features: Tensor,
        gallery_labels: Tensor,
        top_k: List[int] = None,
    ) -> Dict[str, float]:
        """Compute retrieval metrics (Precision@K, Recall@K, MAP).

        Useful for evaluating how well the representation supports
        content-based medical image retrieval.

        Args:
            query_features: Query feature matrix (N_q, D).
            query_labels: Query labels (N_q,).
            gallery_features: Gallery feature matrix (N_g, D).
            gallery_labels: Gallery labels (N_g,).
            top_k: List of K values for Precision@K and Recall@K.

        Returns:
            Dictionary with retrieval metrics.
        """
        if top_k is None:
            top_k = [1, 5, 10, 20]

        if self.config.use_cosine:
            query_features = F.normalize(query_features, dim=1)
            gallery_features = F.normalize(gallery_features, dim=1)

        # Compute pairwise similarity
        sim = torch.mm(
            query_features.to(self.device),
            gallery_features.to(self.device).t(),
        )

        # Sort by similarity (descending)
        _, indices = sim.sort(dim=1, descending=True)

        metrics: Dict[str, float] = {}
        gallery_labels = gallery_labels.to(self.device)
        query_labels = query_labels.to(self.device)

        for k in top_k:
            top_k_indices = indices[:, :k]
            top_k_labels = gallery_labels[top_k_indices]

            # Precision@K
            matches = (top_k_labels == query_labels.unsqueeze(1)).float()
            precision = matches.sum(dim=1) / k
            metrics[f"precision@{k}"] = precision.mean().item()

            # Recall@K
            num_relevant = (gallery_labels.unsqueeze(0) == query_labels.unsqueeze(1)).float().sum(dim=1)
            recall = matches.sum(dim=1) / num_relevant.clamp(min=1)
            metrics[f"recall@{k}"] = recall.mean().item()

        logger.info(
            f"Retrieval metrics: " +
            ", ".join(f"{k}={v:.4f}" for k, v in metrics.items())
        )

        return metrics
