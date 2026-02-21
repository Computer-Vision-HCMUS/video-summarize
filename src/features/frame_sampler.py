"""Frame sampling from video for feature extraction."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional

import cv2
import numpy as np


class FrameSampler:
    """
    Sample frames from video at a given rate (frames per second) or at fixed indices.
    """

    def __init__(self, sample_rate: int = 2) -> None:
        """
        Args:
            sample_rate: Target frames per second to sample (e.g. 2 = 2 fps).
        """
        self.sample_rate = sample_rate

    def sample_frames(
        self,
        video_path: str | Path,
        max_frames: Optional[int] = None,
    ) -> Iterator[np.ndarray]:
        """
        Yield RGB frames (H, W, 3) at sample_rate fps.
        """
        path = Path(video_path)
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            frame_interval = max(1, int(round(fps / self.sample_rate)))
            frame_idx = 0
            count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % frame_interval == 0:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    yield frame_rgb
                    count += 1
                    if max_frames is not None and count >= max_frames:
                        break
                frame_idx += 1
        finally:
            cap.release()

    def sample_frames_to_list(
        self,
        video_path: str | Path,
        max_frames: Optional[int] = None,
    ) -> List[np.ndarray]:
        """Sample all frames into a list."""
        return list(self.sample_frames(video_path, max_frames=max_frames))
