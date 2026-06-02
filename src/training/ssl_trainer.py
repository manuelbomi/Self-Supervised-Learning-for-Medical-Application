"""
Unified trainer for self-supervised pretraining on medical images.

Handles the complete training loop for SimCLR, DINO, and MAE methods
with production-grade features:
    - Distributed Data Parallel (DDP) support for multi-GPU training
    - Automatic Mixed Precision (AMP) for memory efficiency
    - LARS / LAMB optimizer support for large-batch stability
    - Cosine learning rate schedule with linear warmup
    - Gradient clipping and accumulation
    - Checkpointing and logging
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from ..methods.base import BaseSSLMethod, SSLConfig

logger = logging.getLogger(__name__)


@dataclass
class TrainerConfig:
    """Configuration for the SSL trainer."""

    # Training
    epochs: int = 200
    batch_size: int = 256
    gradient_accumulation_steps: int = 1

    # Optimizer
    optimizer: str = "adamw"  # adamw, sgd, lars, lamb
    base_lr: float = 1e-4
    weight_decay: float = 0.05
    momentum: float = 0.9
    betas: Tuple[float, float] = (0.9, 0.999)

    # Learning rate schedule
    warmup_epochs: int = 10
    min_lr: float = 1e-6
    lr_schedule: str = "cosine"  # cosine, linear, step

    # Mixed precision
    fp16: bool = True

    # Gradient clipping
    clip_grad: float = 0.0  # 0 = no clipping
    clip_grad_type: str = "norm"  # norm or value

    # Distributed training
    distributed: bool = False
    local_rank: int = 0
    world_size: int = 1

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every: int = 20
    resume_from: Optional[str] = None

    # Logging
    log_every: int = 50
    wandb_project: Optional[str] = None
    wandb_entity: Optional[str] = None


class LARS(torch.optim.Optimizer):
    """Layer-wise Adaptive Rate Scaling (LARS) optimizer.

    Scales the learning rate per-layer based on the ratio of parameter
    norm to gradient norm. Critical for stable large-batch training
    of contrastive SSL methods like SimCLR.

    Reference:
        You, Y. et al. "Large Batch Training of Convolutional Networks." 2017.

    Args:
        params: Model parameters.
        lr: Base learning rate.
        weight_decay: L2 regularization.
        momentum: SGD momentum.
        eta: LARS trust coefficient.
        weight_decay_filter: Function to filter WD-exempt parameters.
        lars_adaptation_filter: Function to filter LARS-exempt parameters.
    """

    def __init__(
        self,
        params,
        lr: float = 0.3,
        weight_decay: float = 1e-6,
        momentum: float = 0.9,
        eta: float = 0.001,
        weight_decay_filter=None,
        lars_adaptation_filter=None,
    ):
        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            eta=eta,
            weight_decay_filter=weight_decay_filter,
            lars_adaptation_filter=lars_adaptation_filter,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                param_norm = torch.norm(p.data)
                grad_norm = torch.norm(grad)

                # Apply weight decay
                wd_filter = group.get("weight_decay_filter")
                if wd_filter is None or not wd_filter(p):
                    if group["weight_decay"] > 0:
                        grad = grad.add(p.data, alpha=group["weight_decay"])

                # LARS scaling
                lars_filter = group.get("lars_adaptation_filter")
                if lars_filter is None or not lars_filter(p):
                    if param_norm > 0 and grad_norm > 0:
                        adaptive_lr = (
                            group["eta"] * param_norm / (grad_norm + 1e-8)
                        )
                        grad = grad * adaptive_lr

                # Momentum
                if group["momentum"] > 0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(p.data)
                    buf = state["momentum_buffer"]
                    buf.mul_(group["momentum"]).add_(grad)
                    grad = buf

                p.data.add_(grad, alpha=-group["lr"])

        return loss


class CosineWarmupScheduler:
    """Cosine annealing learning rate schedule with linear warmup.

    lr(t) = min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(pi * t_cosine / T_cosine))

    where t_cosine = t - warmup_steps, T_cosine = total_steps - warmup_steps.

    Args:
        optimizer: The optimizer to schedule.
        warmup_steps: Number of linear warmup steps.
        total_steps: Total number of training steps.
        min_lr: Minimum learning rate at end of schedule.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]
        self._step = 0

    def step(self) -> None:
        """Update learning rate for the current step."""
        self._step += 1
        lrs = self._get_lrs()
        for group, lr in zip(self.optimizer.param_groups, lrs):
            group["lr"] = lr

    def _get_lrs(self) -> List[float]:
        """Compute learning rates for all parameter groups."""
        if self._step < self.warmup_steps:
            # Linear warmup
            scale = self._step / max(1, self.warmup_steps)
            return [base_lr * scale for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self._step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            cosine_scale = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_scale
                for base_lr in self.base_lrs
            ]

    def get_last_lr(self) -> List[float]:
        return self._get_lrs()


class SSLTrainer:
    """Unified trainer for self-supervised pretraining.

    Manages the complete training loop with support for all SSL methods
    (SimCLR, DINO, MAE). Handles distributed training, mixed precision,
    gradient accumulation, and periodic evaluation.

    Args:
        model: SSL method instance (SimCLR, DINO, or MAE).
        config: Trainer configuration.
    """

    def __init__(self, model: BaseSSLMethod, config: TrainerConfig) -> None:
        self.config = config
        self.device = torch.device(
            f"cuda:{config.local_rank}" if torch.cuda.is_available() else "cpu"
        )

        # Move model to device
        self.model = model.to(self.device)

        # Wrap in DDP for distributed training
        if config.distributed:
            self.model = DDP(
                self.model,
                device_ids=[config.local_rank],
                find_unused_parameters=True,
            )
            self._unwrapped_model = self.model.module
        else:
            self._unwrapped_model = self.model

        # Build optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scaler = GradScaler(enabled=config.fp16)

        # State tracking
        self.start_epoch = 0
        self.global_step = 0
        self._best_loss = float("inf")

        # Resume from checkpoint if specified
        if config.resume_from:
            self._resume_checkpoint(Path(config.resume_from))

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        param_groups = self._unwrapped_model.get_learnable_params()

        # Scale learning rate by batch size (linear scaling rule)
        effective_batch = (
            self.config.batch_size
            * self.config.world_size
            * self.config.gradient_accumulation_steps
        )
        scaled_lr = self.config.base_lr * effective_batch / 256.0

        for group in param_groups:
            lr_scale = group.pop("lr_scale", 1.0)
            group["lr"] = scaled_lr * lr_scale

        if self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                param_groups, lr=scaled_lr, betas=self.config.betas
            )
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(
                param_groups, lr=scaled_lr, momentum=self.config.momentum
            )
        elif self.config.optimizer == "lars":
            return LARS(
                param_groups,
                lr=scaled_lr,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")

    def _build_scheduler(
        self, steps_per_epoch: int
    ) -> CosineWarmupScheduler:
        """Create learning rate scheduler."""
        total_steps = self.config.epochs * steps_per_epoch
        warmup_steps = self.config.warmup_epochs * steps_per_epoch

        return CosineWarmupScheduler(
            optimizer=self.optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr=self.config.min_lr,
        )

    def _resume_checkpoint(self, path: Path) -> None:
        """Resume training from a checkpoint."""
        self.start_epoch = self._unwrapped_model.load_checkpoint(
            path, optimizer=self.optimizer
        )
        logger.info(f"Resumed from epoch {self.start_epoch}")

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, Any]:
        """Run the full pretraining loop.

        Args:
            train_loader: DataLoader for SSL pretraining data.
            val_loader: Optional validation loader for periodic evaluation.

        Returns:
            Training history dictionary.
        """
        steps_per_epoch = len(train_loader) // self.config.gradient_accumulation_steps
        scheduler = self._build_scheduler(steps_per_epoch)

        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        history: Dict[str, List[float]] = {"loss": [], "lr": [], "epoch_time": []}

        logger.info(
            f"Starting SSL pretraining: {self.config.epochs} epochs, "
            f"batch_size={self.config.batch_size}, "
            f"optimizer={self.config.optimizer}, "
            f"fp16={self.config.fp16}"
        )

        for epoch in range(self.start_epoch, self.config.epochs):
            epoch_start = time.time()

            # Update sampler for distributed training
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)

            # Epoch-level hooks (e.g., update EMA schedule for DINO)
            self._unwrapped_model.on_epoch_start(epoch)

            # Training epoch
            epoch_loss = self._train_epoch(
                train_loader, scheduler, epoch
            )

            epoch_time = time.time() - epoch_start
            history["loss"].append(epoch_loss)
            history["lr"].append(scheduler.get_last_lr()[0])
            history["epoch_time"].append(epoch_time)

            # Logging
            if self._is_main_process():
                logger.info(
                    f"Epoch {epoch + 1}/{self.config.epochs} | "
                    f"Loss: {epoch_loss:.4f} | "
                    f"LR: {scheduler.get_last_lr()[0]:.6f} | "
                    f"Time: {epoch_time:.1f}s"
                )

            # Save checkpoint
            if (
                self._is_main_process()
                and (epoch + 1) % self.config.save_every == 0
            ):
                ckpt_path = checkpoint_dir / f"ssl_epoch{epoch + 1:04d}.pth"
                self._unwrapped_model.save_checkpoint(
                    ckpt_path, epoch + 1, self.optimizer
                )

                if epoch_loss < self._best_loss:
                    self._best_loss = epoch_loss
                    best_path = checkpoint_dir / "ssl_best.pth"
                    self._unwrapped_model.save_checkpoint(
                        best_path, epoch + 1, self.optimizer
                    )

        # Save final checkpoint
        if self._is_main_process():
            final_path = checkpoint_dir / "ssl_final.pth"
            self._unwrapped_model.save_checkpoint(
                final_path, self.config.epochs, self.optimizer
            )

        return history

    def _train_epoch(
        self,
        train_loader: DataLoader,
        scheduler: CosineWarmupScheduler,
        epoch: int,
    ) -> float:
        """Run a single training epoch.

        Args:
            train_loader: Training data loader.
            scheduler: Learning rate scheduler.
            epoch: Current epoch index.

        Returns:
            Average loss for the epoch.
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                if isinstance(batch[0], list):
                    # Multi-crop: [[crop1, crop2, ...], labels]
                    images = [crop.to(self.device, non_blocking=True) for crop in batch[0]]
                    labels = batch[1] if len(batch) > 1 else None
                elif isinstance(batch[0], Tensor) and batch[0].dim() == 4:
                    images = batch[0].to(self.device, non_blocking=True)
                    labels = batch[1] if len(batch) > 1 else None
                else:
                    images = [b.to(self.device, non_blocking=True) for b in batch[:2]]
                    labels = batch[2] if len(batch) > 2 else None
            else:
                images = batch.to(self.device, non_blocking=True)
                labels = None

            # Forward pass with mixed precision
            with autocast(enabled=self.config.fp16):
                output = self.model(images)
                loss = output["loss"]
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if self.config.clip_grad > 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.config.clip_grad_type == "norm":
                        nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.config.clip_grad
                        )
                    else:
                        nn.utils.clip_grad_value_(
                            self.model.parameters(), self.config.clip_grad
                        )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Step-level hooks (e.g., EMA update for DINO)
                self._unwrapped_model.on_step_end(self.global_step)

                scheduler.step()
                self.global_step += 1

            total_loss += output["loss"].item()
            num_batches += 1

            # Periodic logging
            if (
                self._is_main_process()
                and batch_idx % self.config.log_every == 0
            ):
                current_lr = scheduler.get_last_lr()[0]
                extra = ""
                if "avg_pos_sim" in output:
                    extra += f" | Pos Sim: {output['avg_pos_sim']:.4f}"
                if "teacher_entropy" in output:
                    extra += f" | T-Entropy: {output['teacher_entropy']:.2f}"

                logger.info(
                    f"  [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {output['loss'].item():.4f} | "
                    f"LR: {current_lr:.6f}{extra}"
                )

        return total_loss / max(num_batches, 1)

    def _is_main_process(self) -> bool:
        """Check if this is the main process (for logging/saving)."""
        return self.config.local_rank == 0


def setup_distributed() -> Tuple[int, int]:
    """Initialize distributed training environment.

    Returns:
        Tuple of (local_rank, world_size).
    """
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()

    torch.cuda.set_device(local_rank)
    logger.info(f"Initialized distributed: rank {local_rank}/{world_size}")

    return local_rank, world_size
