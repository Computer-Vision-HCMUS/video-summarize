"""BiLSTM model for frame-level importance scoring."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .attention import TemporalAttention


class BiLSTMSummarizer(nn.Module):
    """
    BiLSTM that consumes frame features and outputs frame importance scores.
    Optional temporal attention for context; scores are per-frame.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_dim,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        out_h = hidden_size * self.num_directions
        if use_attention:
            self.attention = TemporalAttention(out_h)
        else:
            self.attention = None
        self.scorer = nn.Sequential(
            nn.Linear(out_h, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x: (B, T, D). lengths: (B,) actual lengths.
        Returns (scores (B, T), attention_weights (B, T) or None).
        """
        if lengths is not None:
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            out, _ = pad_packed_sequence(packed_out, batch_first=True)
        else:
            out, _ = self.lstm(x)

        attn_weights = None
        if self.attention is not None:
            _, attn_weights = self.attention(out, lengths=lengths)

        scores = self.scorer(out).squeeze(-1)
        return scores, attn_weights
