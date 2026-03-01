"""
MultimodalFeatureLoader: load và concat visual + audio features.

Drop-in replacement cho FeatureLoader khi có audio features.

Convention file:
    data/features/{video_id}.npy        ← visual [T, 1024]
    data/features/{video_id}_audio.npy  ← audio  [T, 384]

Output: concat [T, 1024+384] = [T, 1408]

Dùng trong train.py:
    # Thay FeatureLoader bằng MultimodalFeatureLoader
    feature_loader = MultimodalFeatureLoader(
        features_root=paths.features_root,
        visual_dim=1024,
        audio_dim=384,
    )
    # input_dim của model = 1408
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


class MultimodalFeatureLoader:
    """
    Load visual + audio features và concat thành multimodal features.

    Backward compatible với FeatureLoader:
      - Nếu audio file không tồn tại → chỉ dùng visual (không crash)
      - Cùng interface: .load(video_id) → Tensor [T, D]

    Args:
        features_root:  Thư mục chứa .npy files.
        visual_dim:     Dimension visual features (1024 GoogLeNet / 2048 ResNet).
        audio_dim:      Dimension audio features (384 SentenceBERT).
        audio_weight:   Weight của audio features khi concat. 1.0 = không scale.
        padding_value:  Giá trị padding.
        nan_fill:       Giá trị fill NaN.
    """

    def __init__(
        self,
        features_root: str | Path,
        visual_dim:    int   = 1024,
        audio_dim:     int   = 384,
        audio_weight:  float = 1.0,
        padding_value: float = 0.0,
        nan_fill:      float = 0.0,
    ) -> None:
        self.features_root = Path(features_root)
        self.visual_dim    = visual_dim
        self.audio_dim     = audio_dim
        self.audio_weight  = audio_weight
        self.padding_value = padding_value
        self.nan_fill      = nan_fill
        # feature_dim sau concat
        self.feature_dim   = visual_dim + audio_dim

    def _load_npy(self, path: Path, expected_dim: int) -> Optional[np.ndarray]:
        """Load .npy file, handle NaN và shape issues."""
        if not path.exists():
            return None
        try:
            arr = np.load(path)
            arr = np.nan_to_num(arr, nan=self.nan_fill, posinf=0.0, neginf=0.0)
            if arr.ndim == 1:
                arr = arr.reshape(-1, expected_dim)
            if arr.shape[-1] != expected_dim:
                logger.warning(
                    "Dim mismatch %s: expected %d, got %d",
                    path.name, expected_dim, arr.shape[-1]
                )
                return None
            return arr.astype(np.float32)
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)
            return None

    def load(self, video_id: str) -> Optional[torch.Tensor]:
        """
        Load visual + audio features → concat → Tensor [T, visual_dim + audio_dim].

        Nếu không có audio features → chỉ dùng visual [T, visual_dim].
        Caller nên kiểm tra tensor.shape[-1] để biết actual feature_dim.
        """
        # Load visual
        visual_path = self.features_root / f"{video_id}.npy"
        visual      = self._load_npy(visual_path, self.visual_dim)
        if visual is None:
            logger.warning("Visual features không tìm thấy: %s", video_id)
            return None

        T = visual.shape[0]

        # Load audio (optional)
        audio_path = self.features_root / f"{video_id}_audio.npy"
        audio      = self._load_npy(audio_path, self.audio_dim)

        if audio is not None:
            # Align length (T có thể lệch nhẹ)
            T_audio = audio.shape[0]
            if T_audio != T:
                if T_audio > T:
                    audio = audio[:T]
                else:
                    # Pad audio nếu ngắn hơn
                    pad   = np.zeros((T - T_audio, self.audio_dim), dtype=np.float32)
                    audio = np.concatenate([audio, pad], axis=0)

            # Scale audio nếu cần
            if self.audio_weight != 1.0:
                audio = audio * self.audio_weight

            # Concat: [T, visual_dim + audio_dim]
            combined = np.concatenate([visual, audio], axis=1)
            logger.debug("Loaded multimodal features for %s: %s", video_id, combined.shape)
        else:
            # Fallback: chỉ dùng visual, pad zeros cho audio
            logger.debug(
                "Audio features không có cho %s. Dùng visual-only + zero audio.", video_id
            )
            zero_audio = np.zeros((T, self.audio_dim), dtype=np.float32)
            combined   = np.concatenate([visual, zero_audio], axis=1)

        return torch.from_numpy(combined).float()

    def has_audio(self, video_id: str) -> bool:
        """Kiểm tra video có audio features không."""
        return (self.features_root / f"{video_id}_audio.npy").exists()

    def exists(self, video_id: str) -> bool:
        """Kiểm tra visual features tồn tại."""
        return (self.features_root / f"{video_id}.npy").exists()

    def audio_coverage(self, video_ids: list[str]) -> float:
        """Tỷ lệ videos có audio features."""
        n = sum(1 for v in video_ids if self.has_audio(v))
        return n / len(video_ids) if video_ids else 0.0
