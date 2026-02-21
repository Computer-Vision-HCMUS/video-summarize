"""Dataset class for video summarization with sequence padding and temporal batching."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from .feature_loader import FeatureLoader
from .label_loader import LabelLoader

logger = logging.getLogger(__name__)


class VideoSummarizationDataset(Dataset):
    """
    Dataset of (features, labels, lengths) for supervised video summarization.
    Handles variable-length sequences via padding and length masks.
    """

    def __init__(
        self,
        video_ids: List[str],
        feature_loader: FeatureLoader,
        label_loader: LabelLoader,
        max_seq_len: int = 320,
        min_seq_len: int = 16,
        padding_value: float = 0.0,
        pad_labels: bool = True,
    ) -> None:
        self.video_ids = video_ids
        self.feature_loader = feature_loader
        self.label_loader = label_loader
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.padding_value = padding_value
        self.pad_labels = pad_labels
        self._valid_indices: Optional[List[int]] = None
        self._build_valid_indices()

    def _build_valid_indices(self) -> None:
        """Precompute indices for videos that have both features and labels."""
        valid = []
        for i, vid in enumerate(self.video_ids):
            feats = self.feature_loader.load(vid)
            labels = self.label_loader.load(vid)
            if feats is None or labels is None:
                continue
            T = min(feats.size(0), labels.size(0))
            if T < self.min_seq_len:
                continue
            valid.append(i)
        self._valid_indices = valid
        logger.info("Dataset: %d valid of %d videos", len(valid), len(self.video_ids))

    def __len__(self) -> int:
        return len(self._valid_indices) if self._valid_indices else 0

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        """
        Returns:
            features: (T_padded, D)
            labels: (T_padded,) or (T_padded,) same length
            length: original length (before padding)
            video_id: str
        """
        real_idx = self._valid_indices[idx]
        video_id = self.video_ids[real_idx]
        feats = self.feature_loader.load(video_id)
        labels = self.label_loader.load(video_id)
        assert feats is not None and labels is not None
        T = min(feats.size(0), labels.size(0))
        T = min(T, self.max_seq_len)
        feats = feats[:T]
        labels = labels[:T]
        length = T

        if T < self.max_seq_len:
            pad_len = self.max_seq_len - T
            feats = torch.nn.functional.pad(
                feats, (0, 0, 0, pad_len), value=self.padding_value
            )
            if self.pad_labels:
                labels = torch.nn.functional.pad(labels, (0, pad_len), value=0.0)

        return feats, labels, length, video_id
