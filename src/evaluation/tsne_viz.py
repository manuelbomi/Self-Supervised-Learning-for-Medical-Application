"""
t-SNE and UMAP visualization of learned SSL embeddings.

Generates 2D visualizations of high-dimensional representations to
qualitatively assess clustering quality, class separation, and the
structure of the learned embedding space.

Good SSL representations should show:
    - Clear separation between classes
    - Tight, well-defined clusters
    - Meaningful sub-cluster structure corresponding to clinical subtypes
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for embedding visualization."""

    # t-SNE parameters
    tsne_perplexity: float = 30.0
    tsne_n_iter: int = 1000
    tsne_learning_rate: float = 200.0
    tsne_early_exaggeration: float = 12.0

    # UMAP parameters
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"

    # Visualization
    max_samples: int = 5000  # Limit for computational tractability
    figure_size: Tuple[int, int] = (10, 8)
    point_size: float = 5.0
    alpha: float = 0.7
    colormap: str = "tab10"
    dpi: int = 150

    # Output
    output_dir: str = "visualizations"
    save_embeddings: bool = True

    # Class names for labels
    class_names: Optional[List[str]] = None


class EmbeddingVisualizer:
    """Generate t-SNE and UMAP visualizations of SSL representations.

    Extracts features from a frozen SSL model, reduces dimensionality,
    and creates publication-quality scatter plots colored by class label.

    Args:
        ssl_model: Pretrained SSL model (frozen for feature extraction).
        config: Visualization configuration.
        device: Compute device.
    """

    def __init__(
        self,
        ssl_model: nn.Module,
        config: VisualizationConfig,
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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract features and labels from dataset.

        Args:
            dataloader: DataLoader yielding (images, labels).

        Returns:
            Tuple of (features_array, labels_array) as numpy arrays.
        """
        self.ssl_model.eval()
        all_features: List[Tensor] = []
        all_labels: List[Tensor] = []
        count = 0

        for images, labels in dataloader:
            if count >= self.config.max_samples:
                break
            images = images.to(self.device)
            features = self.ssl_model.extract_features(images)
            features = F.normalize(features, dim=1)

            all_features.append(features.cpu())
            all_labels.append(labels)
            count += images.shape[0]

        features = torch.cat(all_features, dim=0)[: self.config.max_samples]
        labels = torch.cat(all_labels, dim=0)[: self.config.max_samples]

        return features.numpy(), labels.numpy()

    def compute_tsne(
        self, features: np.ndarray, random_state: int = 42
    ) -> np.ndarray:
        """Compute t-SNE embedding of features.

        Args:
            features: Feature array of shape (N, D).
            random_state: Random seed for reproducibility.

        Returns:
            2D embedding of shape (N, 2).
        """
        from sklearn.manifold import TSNE

        logger.info(
            f"Computing t-SNE (perplexity={self.config.tsne_perplexity}, "
            f"n_iter={self.config.tsne_n_iter})..."
        )

        tsne = TSNE(
            n_components=2,
            perplexity=self.config.tsne_perplexity,
            n_iter=self.config.tsne_n_iter,
            learning_rate=self.config.tsne_learning_rate,
            early_exaggeration=self.config.tsne_early_exaggeration,
            random_state=random_state,
            init="pca",
        )

        embedding = tsne.fit_transform(features)
        logger.info(f"t-SNE complete. KL divergence: {tsne.kl_divergence_:.4f}")

        return embedding

    def compute_umap(
        self, features: np.ndarray, random_state: int = 42
    ) -> np.ndarray:
        """Compute UMAP embedding of features.

        Args:
            features: Feature array of shape (N, D).
            random_state: Random seed for reproducibility.

        Returns:
            2D embedding of shape (N, 2).
        """
        try:
            import umap
        except ImportError:
            logger.warning("umap-learn not installed. Falling back to t-SNE.")
            return self.compute_tsne(features, random_state)

        logger.info(
            f"Computing UMAP (n_neighbors={self.config.umap_n_neighbors}, "
            f"min_dist={self.config.umap_min_dist})..."
        )

        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=self.config.umap_n_neighbors,
            min_dist=self.config.umap_min_dist,
            metric=self.config.umap_metric,
            random_state=random_state,
        )

        embedding = reducer.fit_transform(features)
        logger.info("UMAP complete.")

        return embedding

    def plot_embedding(
        self,
        embedding: np.ndarray,
        labels: np.ndarray,
        title: str = "Embedding Visualization",
        save_path: Optional[Path] = None,
        method_name: str = "t-SNE",
    ) -> None:
        """Create a scatter plot of the 2D embedding.

        Args:
            embedding: 2D embedding array (N, 2).
            labels: Class labels (N,).
            title: Plot title.
            save_path: Path to save the figure.
            method_name: Name of the reduction method (for axis labels).
        """
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=self.config.figure_size)

        unique_labels = np.unique(labels)
        cmap = plt.get_cmap(self.config.colormap)
        colors = [cmap(i / max(len(unique_labels) - 1, 1)) for i in range(len(unique_labels))]

        for i, label in enumerate(unique_labels):
            mask = labels == label
            class_name = (
                self.config.class_names[int(label)]
                if self.config.class_names and int(label) < len(self.config.class_names)
                else f"Class {label}"
            )
            ax.scatter(
                embedding[mask, 0],
                embedding[mask, 1],
                c=[colors[i]],
                label=class_name,
                s=self.config.point_size,
                alpha=self.config.alpha,
                edgecolors="none",
            )

        ax.set_xlabel(f"{method_name} Dimension 1", fontsize=12)
        ax.set_ylabel(f"{method_name} Dimension 2", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.legend(loc="best", fontsize=10, markerscale=3)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=self.config.dpi, bbox_inches="tight")
            logger.info(f"Figure saved to {save_path}")

        plt.close(fig)

    def visualize(
        self,
        dataloader: DataLoader,
        save_dir: Optional[str] = None,
        methods: List[str] = None,
        title_prefix: str = "",
    ) -> Dict[str, np.ndarray]:
        """Run full visualization pipeline: extract, reduce, plot.

        Args:
            dataloader: DataLoader yielding (images, labels).
            save_dir: Directory to save plots and embeddings.
            methods: List of reduction methods to use ("tsne", "umap").
            title_prefix: Prefix for plot titles.

        Returns:
            Dictionary of method_name -> 2D embeddings.
        """
        if methods is None:
            methods = ["tsne"]

        save_dir = Path(save_dir or self.config.output_dir)

        # Extract features
        features, labels = self.extract_features(dataloader)
        logger.info(f"Extracted {features.shape[0]} features of dim {features.shape[1]}")

        # Optionally save raw features
        if self.config.save_embeddings:
            save_dir.mkdir(parents=True, exist_ok=True)
            np.save(save_dir / "features.npy", features)
            np.save(save_dir / "labels.npy", labels)

        embeddings: Dict[str, np.ndarray] = {}

        for method in methods:
            if method == "tsne":
                embedding = self.compute_tsne(features)
                title = f"{title_prefix}t-SNE Visualization"
            elif method == "umap":
                embedding = self.compute_umap(features)
                title = f"{title_prefix}UMAP Visualization"
            else:
                logger.warning(f"Unknown method: {method}")
                continue

            embeddings[method] = embedding

            self.plot_embedding(
                embedding=embedding,
                labels=labels,
                title=title,
                save_path=save_dir / f"{method}_embedding.png",
                method_name=method.upper(),
            )

        return embeddings

    @staticmethod
    def compute_cluster_metrics(
        features: np.ndarray, labels: np.ndarray
    ) -> Dict[str, float]:
        """Compute quantitative clustering metrics on the embedding space.

        Useful for comparing different SSL methods' embedding quality
        beyond visual inspection.

        Args:
            features: Feature array (N, D).
            labels: Class labels (N,).

        Returns:
            Dictionary with silhouette score and other metrics.
        """
        from sklearn.metrics import silhouette_score, calinski_harabasz_score

        metrics: Dict[str, float] = {}

        try:
            metrics["silhouette_score"] = silhouette_score(
                features, labels, metric="cosine", sample_size=min(5000, len(features))
            )
        except ValueError:
            metrics["silhouette_score"] = 0.0

        try:
            metrics["calinski_harabasz"] = calinski_harabasz_score(features, labels)
        except ValueError:
            metrics["calinski_harabasz"] = 0.0

        # Compute inter/intra-class distance ratio
        unique_labels = np.unique(labels)
        intra_dists = []
        centroids = []

        for label in unique_labels:
            mask = labels == label
            class_features = features[mask]
            centroid = class_features.mean(axis=0)
            centroids.append(centroid)
            dists = np.linalg.norm(class_features - centroid, axis=1)
            intra_dists.append(dists.mean())

        centroids = np.stack(centroids)
        inter_dists = []
        for i in range(len(centroids)):
            for j in range(i + 1, len(centroids)):
                inter_dists.append(np.linalg.norm(centroids[i] - centroids[j]))

        avg_intra = np.mean(intra_dists)
        avg_inter = np.mean(inter_dists) if inter_dists else 0.0
        metrics["inter_intra_ratio"] = avg_inter / (avg_intra + 1e-8)

        logger.info(
            f"Cluster metrics: silhouette={metrics['silhouette_score']:.4f}, "
            f"inter/intra={metrics['inter_intra_ratio']:.4f}"
        )

        return metrics
