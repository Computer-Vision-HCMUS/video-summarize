"""Load and cache video features from disk."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class FeatureLoader:
    """Load pre-extracted features; handle NaN and missing files."""

    def __init__(
        self,
        features_root: str | Path,
        feature_dim: int,
        padding_value: float = 0.0,
        nan_fill: float = 0.0,
    ) -> None:
        self.features_root = Path(features_root)
        self.feature_dim = feature_dim
        self.padding_value = padding_value
        self.nan_fill = nan_fill

    def load(self, video_id: str) -> Optional[torch.Tensor]:
        """
        Load features for a video. Returns (T, D) tensor or None if missing.
        Replaces NaN with nan_fill.
        """
        path = self.features_root / f"{video_id}.npy"
        if not path.exists():
            logger.warning("Feature file not found: %s", path)
            return None
        try:
            arr = np.load(path)
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)
            return None
        if arr.size == 0:
            return None
        arr = np.nan_to_num(arr, nan=self.nan_fill, posinf=0.0, neginf=0.0)
        if arr.ndim == 1:
            arr = arr.reshape(-1, self.feature_dim)
        return torch.from_numpy(arr).float()

    def exists(self, video_id: str) -> bool:
        """Check if feature file exists."""
        return (self.features_root / f"{video_id}.npy").exists()
