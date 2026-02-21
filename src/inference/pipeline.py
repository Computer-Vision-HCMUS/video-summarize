"""End-to-end inference: load model, features -> scores -> static/dynamic summary."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from ..data import FeatureLoader
from ..evaluation import select_keyshots
from ..models import BiLSTMSummarizer
from .static_summary import generate_static_summary
from .dynamic_summary import generate_dynamic_summary, export_summary_video

logger = logging.getLogger(__name__)


class InferencePipeline:
    """
    Load model and features; produce frame scores; generate static/dynamic summaries.
    """

    def __init__(
        self,
        model: BiLSTMSummarizer,
        feature_loader: FeatureLoader,
        device: Optional[torch.device] = None,
        summary_ratio: float = 0.15,
        min_keyframes: int = 5,
        skim_fps: int = 15,
    ) -> None:
        self.model = model
        self.feature_loader = feature_loader
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.summary_ratio = summary_ratio
        self.min_keyframes = min_keyframes
        self.skim_fps = skim_fps

    def predict_scores(self, video_id: str) -> Tuple[Optional[np.ndarray], int]:
        """
        Get frame importance scores for a video. Returns (scores (T,), length) or (None, 0).
        """
        feats = self.feature_loader.load(video_id)
        if feats is None:
            return None, 0
        feats = feats.unsqueeze(0).to(self.device)
        length = feats.size(1)
        with torch.no_grad():
            scores, _ = self.model(feats, lengths=torch.tensor([length], device=self.device))
        return scores[0].cpu().numpy(), length

    def select_keyframes(
        self,
        scores: np.ndarray,
        length: int,
    ) -> List[int]:
        """Select keyframe indices from scores."""
        return select_keyshots(
            scores, length,
            summary_ratio=self.summary_ratio,
            min_keyframes=self.min_keyframes,
        )

    def run_static_summary(
        self,
        video_path: Union[str, Path],
        video_id: str,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> List[np.ndarray]:
        """Generate static summary (keyframes) for one video."""
        scores, length = self.predict_scores(video_id)
        if scores is None or length == 0:
            logger.warning("No scores for %s", video_id)
            return []
        keyframes = self.select_keyframes(scores, length)
        return generate_static_summary(
            video_path, keyframes,
            output_dir=output_dir,
            save_images=output_dir is not None,
        )

    def run_dynamic_summary(
        self,
        video_path: Union[str, Path],
        video_id: str,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Optional[Path]:
        """Generate dynamic summary (skim video) and optionally export."""
        scores, length = self.predict_scores(video_id)
        if scores is None or length == 0:
            logger.warning("No scores for %s", video_id)
            return None
        keyframes = self.select_keyframes(scores, length)
        frames = generate_dynamic_summary(video_path, keyframes, target_fps=self.skim_fps)
        if not frames:
            return None
        if output_path is not None:
            return export_summary_video(frames, output_path, fps=self.skim_fps)
        return None
