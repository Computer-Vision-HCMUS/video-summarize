"""Trainer: wires config, data, model, and training loop.

Fix 4: Thay StepLR bằng ReduceLROnPlateau → tự điều chỉnh LR khi val_loss plateau.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from ..config import Config
from ..models import BiLSTMSummarizer
from .loop import run_training_loop

logger = logging.getLogger(__name__)


class Trainer:
    """
    High-level trainer: builds model, optimizer, scheduler, and runs loop.
    """

    def __init__(self, config: Config) -> None:
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    def build_model(self, input_dim: Optional[int] = None) -> BiLSTMSummarizer:
        """Build model from config. input_dim overrides config khi set từ _meta.json."""
        m   = self.config.model
        dim = input_dim if input_dim is not None else m.input_dim
        return BiLSTMSummarizer(
            input_dim=dim,
            hidden_size=m.hidden_size,
            num_layers=m.num_layers,
            dropout=m.dropout,
            bidirectional=getattr(m, "bidirectional", True),
            use_attention=getattr(m, "use_attention", True),
        )

    def run(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        checkpoint_dir: Optional[Path] = None,
        resume_path: Optional[Path] = None,
        model_input_dim: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run training với:
        - ReduceLROnPlateau scheduler (Fix 4)
        - Diversity loss (Fix 3, configured via config)
        """
        model     = self.build_model(input_dim=model_input_dim)
        t         = self.config.training
        criterion = nn.BCEWithLogitsLoss(reduction="mean")

        optimizer = AdamW(
            model.parameters(),
            lr=t.learning_rate,
            weight_decay=t.get("weight_decay", 1e-4),
        )

        # Fix 4: ReduceLROnPlateau thay vì StepLR cứng
        # - patience=5: giảm LR sau 5 epoch không cải thiện val_loss
        # - factor=0.5: giảm LR xuống 50%
        # - min_lr=1e-5: không giảm quá thấp
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
            min_lr=1e-5,
        )

        # Fix 3: diversity loss params từ config (hoặc default)
        diversity_weight = getattr(t, "diversity_weight", 0.3)
        n_segments       = getattr(t, "n_segments", 5)

        logger.info(
            "Training: lr=%.2e | diversity_weight=%.2f | n_segments=%d",
            t.learning_rate, diversity_weight, n_segments,
        )

        return run_training_loop(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=self.device,
            epochs=t.epochs,
            gradient_clip=t.get("gradient_clip"),
            early_stopping_patience=t.early_stopping_patience,
            checkpoint_dir=Path(checkpoint_dir) if checkpoint_dir else None,
            checkpoint_every=t.get("checkpoint_every", 5),
            resume_path=Path(resume_path) if resume_path else None,
            diversity_weight=diversity_weight,
            n_segments=n_segments,
        )
