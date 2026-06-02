#!/usr/bin/env python3
"""
Evaluate self-supervised representations on downstream tasks.

Supports multiple evaluation protocols:
    - Linear probe: train a linear classifier on frozen features
    - k-NN: non-parametric classification using nearest neighbors
    - t-SNE/UMAP: visualization of the learned embedding space

Usage:
    python scripts/evaluate_representations.py \
        --checkpoint checkpoints/dino_ep200.pth \
        --method dino \
        --eval-type linear_probe \
        --data-dir data/mammography \
        --label-file data/mammography/labels.csv

    python scripts/evaluate_representations.py \
        --checkpoint checkpoints/simclr_ep200.pth \
        --method simclr \
        --eval-type knn \
        --k 20

    python scripts/evaluate_representations.py \
        --checkpoint checkpoints/mae_ep400.pth \
        --method mae \
        --eval-type tsne \
        --output-dir visualizations/mae
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.methods.simclr import SimCLR, SimCLRConfig
from src.methods.dino import DINO, DINOConfig
from src.methods.mae import MAE, MAEConfig
from src.backbones.vision_transformer import vit_small, vit_base
from src.backbones.resnet_medical import resnet50_medical
from src.augmentations.medical_augmentations import MedicalAugConfig, build_eval_augmentation
from src.data.medical_dataset import MedicalImageDataset, DatasetConfig
from src.evaluation.linear_probe import LinearProbeEvaluator, LinearProbeConfig
from src.evaluation.knn_eval import KNNEvaluator, KNNConfig
from src.evaluation.tsne_viz import EmbeddingVisualizer, VisualizationConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_ssl_model(
    checkpoint_path: str,
    method: str,
    device: torch.device,
) -> torch.nn.Module:
    """Load a pretrained SSL model from checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file.
        method: SSL method name (simclr, dino, mae).
        device: Device to load the model onto.

    Returns:
        Loaded SSL model in eval mode.
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config_dict = checkpoint.get("config", {})

    embed_dim = config_dict.get("embed_dim", 384)
    in_channels = config_dict.get("in_channels", 1)
    img_size = config_dict.get("img_size", 224)

    # Build backbone based on method
    if method == "simclr":
        backbone = resnet50_medical(in_channels=in_channels)
        ssl_config = SimCLRConfig(embed_dim=backbone.embed_dim, **{
            k: v for k, v in config_dict.items()
            if k in SimCLRConfig.__dataclass_fields__
        })
        model = SimCLR(backbone, ssl_config)
    elif method == "dino":
        backbone = vit_small(in_channels=in_channels, img_size=img_size)
        ssl_config = DINOConfig(embed_dim=embed_dim, **{
            k: v for k, v in config_dict.items()
            if k in DINOConfig.__dataclass_fields__
        })
        model = DINO(backbone, ssl_config)
    elif method == "mae":
        backbone = vit_base(in_channels=in_channels, img_size=img_size)
        ssl_config = MAEConfig(embed_dim=embed_dim, **{
            k: v for k, v in config_dict.items()
            if k in MAEConfig.__dataclass_fields__
        })
        model = MAE(backbone, ssl_config)
    else:
        raise ValueError(f"Unknown method: {method}")

    model.load_state_dict(checkpoint["state_dict"], strict=False)
    model = model.to(device)
    model.eval()

    logger.info(
        f"Loaded {method.upper()} model from {checkpoint_path} "
        f"(epoch {checkpoint.get('epoch', '?')})"
    )
    return model


def run_linear_probe(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    """Run linear probe evaluation.

    Args:
        model: Pretrained SSL model.
        train_loader: Training data loader.
        val_loader: Validation data loader.
        args: Command-line arguments.
        device: Compute device.

    Returns:
        Evaluation results dictionary.
    """
    config = LinearProbeConfig(
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
        lr=args.probe_lr,
        epochs=args.probe_epochs,
        batch_size=args.batch_size,
        use_bn=True,
    )

    evaluator = LinearProbeEvaluator(model, config, device=device)
    results = evaluator.train(train_loader, val_loader)

    logger.info(f"Linear Probe Results:")
    logger.info(f"  Best Accuracy: {results['best_accuracy']:.4f}")
    logger.info(f"  Best Epoch: {results['best_epoch'] + 1}")

    return results


def run_knn_eval(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    """Run k-NN evaluation.

    Args:
        model: Pretrained SSL model.
        train_loader: Training data (memory bank).
        test_loader: Test data (queries).
        args: Command-line arguments.
        device: Compute device.

    Returns:
        Evaluation results dictionary.
    """
    k_values = [int(k) for k in args.k.split(",")] if args.k else [1, 5, 10, 20]

    config = KNNConfig(
        k_values=k_values,
        temperature=0.07,
        num_classes=args.num_classes,
        use_cosine=True,
    )

    evaluator = KNNEvaluator(model, config, device=device)
    results = evaluator.evaluate(train_loader, test_loader)

    logger.info(f"k-NN Results:")
    for k, acc in results["results"].items():
        logger.info(f"  k={k}: accuracy = {acc:.4f}")
    logger.info(f"  Best: k={results['best_k']}, accuracy={results['best_accuracy']:.4f}")

    return results


def run_tsne_visualization(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, Any]:
    """Run t-SNE/UMAP visualization.

    Args:
        model: Pretrained SSL model.
        dataloader: Data loader for feature extraction.
        args: Command-line arguments.
        device: Compute device.

    Returns:
        Dictionary with embeddings and metrics.
    """
    config = VisualizationConfig(
        max_samples=args.max_samples,
        tsne_perplexity=args.perplexity,
        output_dir=args.output_dir,
        class_names=args.class_names.split(",") if args.class_names else None,
    )

    visualizer = EmbeddingVisualizer(model, config, device=device)

    methods = args.viz_methods.split(",") if args.viz_methods else ["tsne"]
    embeddings = visualizer.visualize(
        dataloader,
        save_dir=args.output_dir,
        methods=methods,
        title_prefix=f"{args.method.upper()} - ",
    )

    # Compute cluster metrics
    features, labels = visualizer.extract_features(dataloader)
    metrics = EmbeddingVisualizer.compute_cluster_metrics(features, labels)
    logger.info(f"Cluster Metrics: {metrics}")

    return {"embeddings": embeddings, "cluster_metrics": metrics}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate SSL representations")

    # Required
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--method", type=str, required=True,
                        choices=["simclr", "dino", "mae"])
    parser.add_argument("--eval-type", type=str, required=True,
                        choices=["linear_probe", "knn", "tsne", "all"])

    # Data
    parser.add_argument("--data-dir", type=str, default="data/mammography")
    parser.add_argument("--label-file", type=str, default=None)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)

    # Model
    parser.add_argument("--embed-dim", type=int, default=384)
    parser.add_argument("--num-classes", type=int, default=2)

    # Linear probe
    parser.add_argument("--probe-lr", type=float, default=0.1)
    parser.add_argument("--probe-epochs", type=int, default=100)

    # k-NN
    parser.add_argument("--k", type=str, default="1,5,10,20,50,100,200")

    # Visualization
    parser.add_argument("--output-dir", type=str, default="visualizations")
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--viz-methods", type=str, default="tsne")
    parser.add_argument("--class-names", type=str, default=None)

    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    """Main evaluation entry point."""
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load model
    model = load_ssl_model(args.checkpoint, args.method, device)

    # Build evaluation transform (deterministic)
    aug_config = MedicalAugConfig(img_size=args.img_size)
    eval_transform = build_eval_augmentation(aug_config)

    # Build datasets
    dataset_config = DatasetConfig(
        data_dir=args.data_dir,
        img_size=args.img_size,
        label_file=args.label_file,
        num_workers=args.num_workers,
    )

    dataset = MedicalImageDataset(dataset_config, transform=eval_transform)

    # Split into train/val if needed
    n_total = len(dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers,
    )

    # Run evaluation
    eval_types = ["linear_probe", "knn", "tsne"] if args.eval_type == "all" else [args.eval_type]

    all_results = {}
    for eval_type in eval_types:
        logger.info(f"\n{'='*60}")
        logger.info(f"Running {eval_type} evaluation...")
        logger.info(f"{'='*60}")

        if eval_type == "linear_probe":
            all_results["linear_probe"] = run_linear_probe(
                model, train_loader, val_loader, args, device
            )
        elif eval_type == "knn":
            all_results["knn"] = run_knn_eval(
                model, train_loader, val_loader, args, device
            )
        elif eval_type == "tsne":
            full_loader = torch.utils.data.DataLoader(
                dataset, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
            )
            all_results["tsne"] = run_tsne_visualization(
                model, full_loader, args, device
            )

    logger.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
