#!/usr/bin/env python3
"""
Main pretraining script for self-supervised learning on medical images.

Supports SimCLR, DINO, and MAE methods with configurable backbones,
augmentations, and training hyperparameters. Configuration is loaded
from YAML files with command-line overrides.

Usage:
    # Single GPU
    python scripts/pretrain.py --config configs/simclr_mammography.yaml

    # Multi-GPU (DDP)
    torchrun --nproc_per_node=4 scripts/pretrain.py --config configs/dino_mammography.yaml

    # With overrides
    python scripts/pretrain.py --config configs/mae_mammography.yaml \
        --batch-size 128 --epochs 100 --fp16
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.methods.simclr import SimCLR, SimCLRConfig
from src.methods.dino import DINO, DINOConfig
from src.methods.mae import MAE, MAEConfig
from src.backbones.vision_transformer import vit_tiny, vit_small, vit_base
from src.backbones.resnet_medical import resnet18_medical, resnet50_medical
from src.augmentations.medical_augmentations import (
    MedicalAugConfig,
    build_simclr_augmentation,
    build_dino_multicrop,
    build_mae_augmentation,
    DualViewTransform,
    MultiCropWrapper,
)
from src.data.medical_dataset import MedicalImageDataset, DatasetConfig
from src.training.ssl_trainer import SSLTrainer, TrainerConfig, setup_distributed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


BACKBONE_REGISTRY = {
    "vit_tiny": vit_tiny,
    "vit_small": vit_small,
    "vit_base": vit_base,
    "resnet18_medical": resnet18_medical,
    "resnet50_medical": resnet50_medical,
}


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded config from {config_path}")
    return config


def build_backbone(config: Dict[str, Any]) -> torch.nn.Module:
    """Instantiate the backbone network from config.

    Args:
        config: Backbone configuration dictionary.

    Returns:
        Backbone module.
    """
    name = config.pop("name", "vit_small")
    if name not in BACKBONE_REGISTRY:
        raise ValueError(
            f"Unknown backbone '{name}'. Available: {list(BACKBONE_REGISTRY.keys())}"
        )

    backbone = BACKBONE_REGISTRY[name](**config)
    num_params = sum(p.numel() for p in backbone.parameters()) / 1e6
    logger.info(f"Built backbone: {name} ({num_params:.1f}M params)")
    return backbone


def build_ssl_method(
    method_name: str, backbone: torch.nn.Module, config: Dict[str, Any]
) -> torch.nn.Module:
    """Instantiate the SSL method from config.

    Args:
        method_name: One of 'simclr', 'dino', 'mae'.
        backbone: Pretrained or randomly initialized backbone.
        config: Full configuration dictionary.

    Returns:
        SSL method instance.
    """
    backbone_cfg = config.get("backbone", {})
    training_cfg = config.get("training", {})

    if method_name == "simclr":
        method_cfg = config.get("simclr", {})
        ssl_config = SimCLRConfig(
            backbone=backbone_cfg.get("name", "resnet50_medical"),
            embed_dim=backbone_cfg.get("embed_dim", 2048),
            img_size=config["data"]["img_size"],
            in_channels=config["data"]["in_channels"],
            epochs=training_cfg.get("epochs", 200),
            batch_size=training_cfg.get("batch_size", 512),
            base_lr=training_cfg.get("base_lr", 0.3),
            optimizer=training_cfg.get("optimizer", "lars"),
            warmup_epochs=training_cfg.get("warmup_epochs", 10),
            **method_cfg,
        )
        model = SimCLR(backbone, ssl_config)

    elif method_name == "dino":
        method_cfg = config.get("dino", {})
        ssl_config = DINOConfig(
            backbone=backbone_cfg.get("name", "vit_small"),
            embed_dim=backbone_cfg.get("embed_dim", 384),
            img_size=config["data"]["img_size"],
            in_channels=config["data"]["in_channels"],
            epochs=training_cfg.get("epochs", 300),
            batch_size=training_cfg.get("batch_size", 256),
            base_lr=training_cfg.get("base_lr", 5e-4),
            optimizer=training_cfg.get("optimizer", "adamw"),
            warmup_epochs=training_cfg.get("warmup_epochs", 10),
            **method_cfg,
        )
        model = DINO(backbone, ssl_config)

    elif method_name == "mae":
        method_cfg = config.get("mae", {})
        ssl_config = MAEConfig(
            backbone=backbone_cfg.get("name", "vit_base"),
            embed_dim=backbone_cfg.get("embed_dim", 768),
            img_size=config["data"]["img_size"],
            in_channels=config["data"]["in_channels"],
            epochs=training_cfg.get("epochs", 400),
            batch_size=training_cfg.get("batch_size", 512),
            base_lr=training_cfg.get("base_lr", 1.5e-4),
            optimizer=training_cfg.get("optimizer", "adamw"),
            warmup_epochs=training_cfg.get("warmup_epochs", 40),
            **method_cfg,
        )
        model = MAE(backbone, ssl_config)

    else:
        raise ValueError(f"Unknown method: {method_name}")

    logger.info(f"Built SSL method: {model}")
    return model


def build_transforms(
    method_name: str, config: Dict[str, Any]
):
    """Build augmentation transforms for the specified SSL method.

    Args:
        method_name: SSL method name.
        config: Augmentation configuration.

    Returns:
        Transform callable.
    """
    aug_cfg = config.get("augmentations", {})
    data_cfg = config.get("data", {})

    aug_config = MedicalAugConfig(
        img_size=data_cfg.get("img_size", 224),
        in_channels=data_cfg.get("in_channels", 1),
        **aug_cfg,
    )

    if method_name == "simclr":
        base_transform = build_simclr_augmentation(aug_config)
        return DualViewTransform(base_transform)

    elif method_name == "dino":
        global_transforms, local_transforms = build_dino_multicrop(aug_config)
        return MultiCropWrapper(global_transforms, local_transforms)

    elif method_name == "mae":
        return build_mae_augmentation(aug_config)

    else:
        raise ValueError(f"Unknown method: {method_name}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Self-supervised pretraining for medical images"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument("--method", type=str, default=None,
                        help="Override SSL method (simclr, dino, mae)")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--fp16", action="store_true", default=None)
    parser.add_argument("--no-fp16", action="store_false", dest="fp16")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def main() -> None:
    """Main pretraining entry point."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Load config
    config = load_config(args.config)

    # Apply CLI overrides
    method_name = args.method or config.get("method", "simclr")
    if args.batch_size:
        config.setdefault("training", {})["batch_size"] = args.batch_size
    if args.epochs:
        config.setdefault("training", {})["epochs"] = args.epochs
    if args.lr:
        config.setdefault("training", {})["base_lr"] = args.lr
    if args.data_dir:
        config.setdefault("data", {})["data_dir"] = args.data_dir

    # Check for distributed training
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    local_rank = 0
    world_size = 1

    if distributed:
        local_rank, world_size = setup_distributed()
        logger.info(f"Distributed training: rank {local_rank}/{world_size}")

    # Build components
    backbone_config = dict(config.get("backbone", {}))
    backbone = build_backbone(backbone_config)

    model = build_ssl_method(method_name, backbone, config)
    transform = build_transforms(method_name, config)

    # Build dataset and dataloader
    data_cfg = config.get("data", {})
    dataset_config = DatasetConfig(
        data_dir=data_cfg.get("data_dir", "data"),
        img_size=data_cfg.get("img_size", 224),
        in_channels=data_cfg.get("in_channels", 1),
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=data_cfg.get("pin_memory", True),
    )

    dataset = MedicalImageDataset(dataset_config, transform=transform)

    sampler = None
    if distributed:
        from torch.utils.data import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=True)

    training_cfg = config.get("training", {})
    batch_size = training_cfg.get("batch_size", 256)

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=data_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
        persistent_workers=data_cfg.get("num_workers", 4) > 0,
    )

    # Build trainer
    checkpoint_cfg = config.get("checkpoint", {})
    trainer_config = TrainerConfig(
        epochs=training_cfg.get("epochs", 200),
        batch_size=batch_size,
        optimizer=training_cfg.get("optimizer", "adamw"),
        base_lr=training_cfg.get("base_lr", 1e-4),
        weight_decay=training_cfg.get("weight_decay", 0.05),
        warmup_epochs=training_cfg.get("warmup_epochs", 10),
        min_lr=training_cfg.get("min_lr", 1e-6),
        fp16=training_cfg.get("fp16", True),
        clip_grad=training_cfg.get("clip_grad", 0.0),
        distributed=distributed,
        local_rank=local_rank,
        world_size=world_size,
        checkpoint_dir=args.checkpoint_dir or checkpoint_cfg.get("dir", "checkpoints"),
        save_every=checkpoint_cfg.get("save_every", 20),
        resume_from=args.resume or training_cfg.get("resume_from"),
        gradient_accumulation_steps=training_cfg.get("gradient_accumulation_steps", 1),
    )

    trainer = SSLTrainer(model, trainer_config)

    # Run pretraining
    logger.info(f"Starting {method_name.upper()} pretraining...")
    history = trainer.train(train_loader)

    logger.info("Pretraining complete!")
    logger.info(f"Final loss: {history['loss'][-1]:.4f}" if history["loss"] else "No training performed")


if __name__ == "__main__":
    main()
