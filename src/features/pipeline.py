"""Feature extraction pipeline: sample frames, extract CNN features, save to disk."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from .frame_sampler import FrameSampler
from .cnn_extractor import CNNFeatureExtractor

logger = logging.getLogger(__name__)


class FeatureExtractionPipeline:
    """
    End-to-end pipeline: video -> sampled frames -> CNN features -> .npy on disk.
    """

    def __init__(
        self,
        output_dir: str | Path,
        frame_sampler: FrameSampler,
        extractor: CNNFeatureExtractor,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_sampler = frame_sampler
        self.extractor = extractor

    def process_video(
        self,
        video_path: str | Path,
        video_id: str,
        max_frames: Optional[int] = None,
        overwrite: bool = False,
    ) -> Optional[Path]:
        """
        Extract features for one video and save to output_dir/{video_id}.npy.
        Returns path to saved file or None on failure.
        """
        out_path = self.output_dir / f"{video_id}.npy"
        if out_path.exists() and not overwrite:
            logger.debug("Skipping (exists): %s", video_id)
            return out_path
        try:
            frames = self.frame_sampler.sample_frames_to_list(video_path, max_frames=max_frames)
            if not frames:
                logger.warning("No frames for %s", video_id)
                return None
            feats = self.extractor.extract_from_frames(frames)
            np.save(out_path, feats)
            logger.info("Saved %s: shape %s", video_id, feats.shape)
            return out_path
        except Exception as e:
            logger.exception("Failed %s: %s", video_id, e)
            return None

    def process_videos(
        self,
        video_paths: List[Path],
        video_ids: Optional[List[str]] = None,
        max_frames_per_video: Optional[int] = None,
        overwrite: bool = False,
    ) -> List[Path]:
        """Process multiple videos. video_ids default to path.stem."""
        if video_ids is None:
            video_ids = [p.stem for p in video_paths]
        if len(video_ids) != len(video_paths):
            raise ValueError("video_paths and video_ids length mismatch")
        saved = []
        for path, vid in zip(video_paths, video_ids):
            p = self.process_video(path, vid, max_frames=max_frames_per_video, overwrite=overwrite)
            if p is not None:
                saved.append(p)
        return saved
