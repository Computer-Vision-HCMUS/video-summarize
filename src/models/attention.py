"""Temporal attention module for frame importance weighting."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Simple additive attention over sequence dimension.
    Input (B, T, H) -> context (B, H), weights (B, T).
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1)

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, T, H). lengths: (B,) optional. mask: (B, T) optional, 1 = valid.
        Returns (context (B, H), weights (B, T)).
        """
        scores = self.attn(x).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        elif lengths is not None:
            B, T = x.size(0), x.size(1)
            mask = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            scores = scores.masked_fill(~mask, -1e9)
        weights = F.softmax(scores, dim=1)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return context, weights
