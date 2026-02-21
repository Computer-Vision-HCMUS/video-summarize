"""Training and validation loops with early stopping and checkpointing."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from ..models import BiLSTMSummarizer

logger = logging.getLogger(__name__)


def run_training_loop(
    model: BiLSTMSummarizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler] = None,
    device: torch.device = None,
    epochs: int = 50,
    gradient_clip: Optional[float] = 1.0,
    early_stopping_patience: int = 10,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_every: int = 5,
    resume_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Full training with validation, early stopping, LR scheduler, checkpointing.
    Returns dict with best_epoch, best_val_loss, history.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    history: Dict[str, list] = {"train_loss": [], "val_loss": []}

    start_epoch = 0
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        logger.info("Resumed from epoch %s", start_epoch)

    for epoch in range(start_epoch, epochs):
        train_loss = _train_epoch(
            model, train_loader, criterion, optimizer, device, gradient_clip
        )
        val_loss = _validate_epoch(model, val_loader, criterion, device)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if scheduler is not None:
            scheduler.step()

        logger.info(
            "Epoch %d | train_loss=%.4f | val_loss=%.4f",
            epoch + 1, train_loss, val_loss,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = 0
            if checkpoint_dir:
                checkpoint_dir = Path(checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                _save_checkpoint(
                    model, optimizer, scheduler, epoch, best_val_loss,
                    checkpoint_dir / "best.pt",
                )
        else:
            patience_counter += 1

        if checkpoint_dir and (epoch + 1) % checkpoint_every == 0:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
            _save_checkpoint(
                model, optimizer, scheduler, epoch, best_val_loss,
                Path(checkpoint_dir) / f"epoch_{epoch + 1}.pt",
            )

        if patience_counter >= early_stopping_patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }


def _masked_bce_loss(scores: torch.Tensor, labels: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """BCE loss only on valid positions (first lengths[i] per sample)."""
    total_loss = 0.0
    total_count = 0
    T = min(scores.size(1), labels.size(1))
    for i in range(scores.size(0)):
        L = min(lengths[i].item(), T)
        if L == 0:
            continue
        s = scores[i, :L]
        t = labels[i, :L]
        total_loss += torch.nn.functional.binary_cross_entropy_with_logits(s, t, reduction="sum")
        total_count += L
    return total_loss / total_count if total_count else scores.new_zeros(1)


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    gradient_clip: Optional[float],
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        feats, labels, lengths, _ = batch
        feats = feats.to(device)
        labels = labels.to(device)
        lengths = lengths.to(device)
        optimizer.zero_grad()
        scores, _ = model(feats, lengths)
        loss = _masked_bce_loss(scores, labels, lengths)
        loss.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        total_loss += loss.item() * feats.size(0)
        n += feats.size(0)
    return total_loss / n if n else 0.0


def _validate_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            feats, labels, lengths, _ = batch
            feats = feats.to(device)
            labels = labels.to(device)
            lengths = lengths.to(device)
            scores, _ = model(feats, lengths)
            loss = _masked_bce_loss(scores, labels, lengths)
            total_loss += loss.item() * feats.size(0)
            n += feats.size(0)
    return total_loss / n if n else 0.0


def _save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: Optional[_LRScheduler],
    epoch: int,
    best_val_loss: float,
    path: Path,
) -> None:
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss": best_val_loss,
    }
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(ckpt, path)
    logger.info("Checkpoint saved: %s", path)
