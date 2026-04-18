"""Training and validation loops with early stopping and checkpointing.

Fix 3: Thêm diversity loss — penalty khi model bỏ sót các phần video.
Fix 4: ReduceLROnPlateau thay vì StepLR cứng → fine-tune tốt hơn cuối training.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau, _LRScheduler
from torch.utils.data import DataLoader

from ..models import BiLSTMSummarizer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Fix 3: Diversity Loss
# ─────────────────────────────────────────────────────────────

def _coverage_diversity_loss(
    scores: torch.Tensor,
    lengths: torch.Tensor,
    n_segments: int = 5,
    coverage_weight: float = 0.3,
) -> torch.Tensor:
    """
    Diversity loss: penalty khi model tập trung quá vào 1 đoạn video.

    Chia video thành n_segments phần, tính max predicted score mỗi phần.
    Penalty = variance của max scores → cao khi 1 phần score cao, phần khác thấp.
    → Model học để distribute attention đều hơn, cover đủ nội dung.

    Args:
        scores:          (B, T) predicted scores (logits).
        lengths:         (B,) actual lengths.
        n_segments:      Số segments chia video thành.
        coverage_weight: Weight của diversity loss.

    Returns:
        Scalar diversity loss.
    """
    B = scores.size(0)
    total_div = scores.new_zeros(1)
    count = 0

    for i in range(B):
        L = int(lengths[i].item())
        if L < n_segments * 2:
            continue

        s = torch.sigmoid(scores[i, :L])  # (L,)

        # Chia thành n_segments phần, tính max của mỗi phần
        seg_maxes = []
        seg_size  = L // n_segments
        for j in range(n_segments):
            seg_start = j * seg_size
            seg_end   = (j + 1) * seg_size if j < n_segments - 1 else L
            seg_max   = s[seg_start:seg_end].max()
            seg_maxes.append(seg_max)

        seg_maxes = torch.stack(seg_maxes)

        # Variance của max scores qua các segments
        # Cao → model tập trung vào 1 đoạn (bad)
        # Thấp → coverage đều (good)
        variance = seg_maxes.var()
        total_div = total_div + variance
        count += 1

    if count == 0:
        return scores.new_zeros(1)

    return coverage_weight * (total_div / count)


# ─────────────────────────────────────────────────────────────
# Masked BCE Loss (giữ nguyên từ bản gốc)
# ─────────────────────────────────────────────────────────────

def _masked_bce_loss(
    scores: torch.Tensor,
    labels: torch.Tensor,
    lengths: torch.Tensor,
) -> torch.Tensor:
    """BCE loss chỉ trên valid positions."""
    total_loss = 0.0
    total_count = 0
    T = min(scores.size(1), labels.size(1))
    for i in range(scores.size(0)):
        L = min(int(lengths[i].item()), T)
        if L == 0:
            continue
        s = scores[i, :L]
        t = labels[i, :L]
        total_loss += torch.nn.functional.binary_cross_entropy_with_logits(
            s, t, reduction="sum"
        )
        total_count += L
    return total_loss / total_count if total_count else scores.new_zeros(1)


# ─────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────

def run_training_loop(
    model: BiLSTMSummarizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    scheduler=None,           # ReduceLROnPlateau hoặc _LRScheduler
    device: torch.device = None,
    epochs: int = 50,
    gradient_clip: Optional[float] = 1.0,
    early_stopping_patience: int = 10,
    checkpoint_dir: Optional[Path] = None,
    checkpoint_every: int = 5,
    resume_path: Optional[Path] = None,
    # Fix 3 params
    diversity_weight: float = 0.3,
    n_segments: int = 5,
) -> Dict[str, Any]:
    """
    Full training với:
    - Masked BCE loss
    - Diversity/coverage loss (Fix 3)
    - ReduceLROnPlateau support (Fix 4)
    - Early stopping, checkpointing
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    best_val_loss   = float("inf")
    best_epoch      = 0
    patience_counter = 0
    history: Dict[str, list] = {
        "train_loss": [], "val_loss": [],
        "train_bce": [], "train_div": [],
        "lr": [],
    }

    start_epoch = 0
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt and scheduler is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        logger.info("Resumed from epoch %s", start_epoch)

    for epoch in range(start_epoch, epochs):
        train_loss, train_bce, train_div = _train_epoch(
            model, train_loader, optimizer, device, gradient_clip,
            diversity_weight, n_segments,
        )
        val_loss = _validate_epoch(model, val_loader, device, diversity_weight, n_segments)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_bce"].append(train_bce)
        history["train_div"].append(train_div)

        # Fix 4: ReduceLROnPlateau nhận val_loss, StepLR không nhận argument
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        logger.info(
            "Epoch %d | train=%.4f (bce=%.4f div=%.4f) | val=%.4f | lr=%.2e",
            epoch + 1, train_loss, train_bce, train_div, val_loss, current_lr,
        )

        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_epoch       = epoch + 1
            patience_counter = 0
            if checkpoint_dir:
                Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
                _save_checkpoint(
                    model, optimizer, scheduler, epoch, best_val_loss,
                    Path(checkpoint_dir) / "best.pt",
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
        "best_epoch":    best_epoch,
        "best_val_loss": best_val_loss,
        "history":       history,
    }


def _train_epoch(
    model, loader, optimizer, device, gradient_clip,
    diversity_weight, n_segments,
):
    model.train()
    total_loss = total_bce = total_div = 0.0
    n = 0
    for batch in loader:
        feats, labels, lengths, _ = batch
        feats   = feats.to(device)
        labels  = labels.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()
        scores, _ = model(feats, lengths)

        bce_loss = _masked_bce_loss(scores, labels, lengths)
        div_loss = _coverage_diversity_loss(scores, lengths, n_segments, diversity_weight)
        loss     = bce_loss + div_loss

        loss.backward()
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()

        B             = feats.size(0)
        total_loss   += loss.item()     * B
        total_bce    += bce_loss.item() * B
        total_div    += div_loss.item() * B
        n            += B

    if n == 0:
        return 0.0, 0.0, 0.0
    return total_loss / n, total_bce / n, total_div / n


def _validate_epoch(model, loader, device, diversity_weight, n_segments):
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for batch in loader:
            feats, labels, lengths, _ = batch
            feats   = feats.to(device)
            labels  = labels.to(device)
            lengths = lengths.to(device)
            scores, _ = model(feats, lengths)
            bce = _masked_bce_loss(scores, labels, lengths)
            div = _coverage_diversity_loss(scores, lengths, n_segments, diversity_weight)
            total_loss += (bce + div).item() * feats.size(0)
            n          += feats.size(0)
    return total_loss / n if n else 0.0


def _save_checkpoint(model, optimizer, scheduler, epoch, best_val_loss, path):
    ckpt = {
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "best_val_loss":        best_val_loss,
    }
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    torch.save(ckpt, path)
    logger.info("Checkpoint saved: %s", path)
