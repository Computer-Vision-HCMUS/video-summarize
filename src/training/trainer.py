"""Trainer: wires config, data, model, and training loop."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
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
        """Build model from config. input_dim overrides config when set (e.g. from data/features/_meta.json)."""
        m = self.config.model
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
        Run training. Uses BCEWithLogitsLoss by default for frame scores.
        model_input_dim: override from data/features/_meta.json when using eccv16 .h5 features.
        """
        model = self.build_model(input_dim=model_input_dim)
        t = self.config.training
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
        optimizer = AdamW(
            model.parameters(),
            lr=t.learning_rate,
            weight_decay=t.get("weight_decay", 0),
        )
        scheduler = StepLR(optimizer, step_size=15, gamma=0.5)

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
        )
