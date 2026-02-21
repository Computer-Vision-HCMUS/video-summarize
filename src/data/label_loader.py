"""Load frame-level or shot-level importance labels."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class LabelLoader:
    """
    Load labels from disk. Expects JSON: {"scores": [0.2, 0.8, ...]} or
    {"keyframes": [0, 5, 10]} which is converted to binary scores.
    """

    def __init__(self, labels_root: str | Path) -> None:
        self.labels_root = Path(labels_root)

    def load(self, video_id: str) -> Optional[torch.Tensor]:
        """
        Load labels for a video. Returns (T,) float tensor in [0, 1]
        or None if missing.
        """
        path = self.labels_root / f"{video_id}.json"
        if not path.exists():
            logger.warning("Label file not found: %s", path)
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)
            return None
        if "scores" in data:
            scores = np.array(data["scores"], dtype=np.float32)
        elif "keyframes" in data:
            indices = data["keyframes"]
            length = max(indices) + 1 if indices else 0
            scores = np.zeros(length, dtype=np.float32)
            for i in indices:
                if i < length:
                    scores[i] = 1.0
        else:
            logger.warning("Unknown label format in %s", path)
            return None
        scores = np.nan_to_num(scores, nan=0.0)
        return torch.from_numpy(scores).float()
