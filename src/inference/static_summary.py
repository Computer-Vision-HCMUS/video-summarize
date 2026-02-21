"""Static summary: keyframes as list of images or paths."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np


def generate_static_summary(
    video_path: Union[str, Path],
    keyframe_indices: List[int],
    frame_timestamps: Optional[List[float]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    save_images: bool = True,
) -> List[np.ndarray]:
    """
    Extract keyframes from video. If output_dir and save_images, also save as images.
    keyframe_indices: frame indices (0-based). frame_timestamps: optional time for each frame.
    Returns list of RGB frames (H, W, 3).
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    keyframes = []
    indices_set = set(keyframe_indices)
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in indices_set:
                keyframes.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_idx += 1
    finally:
        cap.release()

    if output_dir and save_images:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for i, kf in enumerate(keyframes):
            out_path = output_dir / f"keyframe_{i:04d}.jpg"
            cv2.imwrite(str(out_path), cv2.cvtColor(kf, cv2.COLOR_RGB2BGR))
    return keyframes
