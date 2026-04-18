"""BiLSTM model for frame-level importance scoring."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from .attention import TemporalAttention


class BiLSTMSummarizer(nn.Module):
    """
    BiLSTM that consumes frame features and outputs frame importance scores.

    Fix 1: Attention context được fuse vào scorer thay vì bỏ phí.
    Mỗi frame score = f(lstm_out[t], attn_context) thay vì chỉ f(lstm_out[t]).
    → Model biết frame nào quan trọng hơn trong ngữ cảnh toàn video.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        bidirectional: bool = True,
        use_attention: bool = True,
        fuse_attention_context: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.fuse_attention_context = fuse_attention_context
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
            # Mặc định (optimize): concat context toàn video vào scorer.
            # fuse_attention_context=False: checkpoint cũ — attention chỉ để trả weights, scorer chỉ nhận lstm out.
            scorer_in = out_h * 2 if fuse_attention_context else out_h
        else:
            self.attention = None
            scorer_in = out_h

        self.scorer = nn.Sequential(
            nn.Linear(scorer_in, hidden_size),
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
            context, attn_weights = self.attention(out, lengths=lengths)
            if self.fuse_attention_context:
                T = out.size(1)
                context_expanded = context.unsqueeze(1).expand(-1, T, -1)
                scorer_input = torch.cat([out, context_expanded], dim=-1)
            else:
                scorer_input = out
        else:
            scorer_input = out

        scores = self.scorer(scorer_input).squeeze(-1)  # (B, T)
        return scores, attn_weights

    @staticmethod
    def infer_attention_flags_from_state_dict(
        state_dict: Dict[str, torch.Tensor],
        hidden_size: int,
        bidirectional: bool,
    ) -> Tuple[bool, bool]:
        """
        Suy ra (use_attention, fuse_attention_context) từ shape `scorer.0.weight`
        và sự có mặt của module `attention.*` — khớp checkpoint cũ (512) vs optimize (1024).
        """
        num_directions = 2 if bidirectional else 1
        out_h = hidden_size * num_directions
        w = state_dict["scorer.0.weight"]
        in_f = int(w.shape[1])
        has_attn = any(k.startswith("attention.") for k in state_dict)
        if in_f == 2 * out_h:
            return True, True
        if in_f == out_h:
            if has_attn:
                return True, False
            return False, False
        raise ValueError(
            f"Unsupported scorer input dim {in_f} (expected {out_h} or {2 * out_h}); "
            "hidden_size/bidirectional may not match checkpoint."
        )
