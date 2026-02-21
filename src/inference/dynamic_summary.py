"""Dynamic summary: video skim and export."""

from __future__ import annotations

from pathlib import Path
from typing import List, Union

import cv2
import numpy as np


def generate_dynamic_summary(
    video_path: Union[str, Path],
    keyframe_indices: List[int],
    target_fps: int = 15,
) -> List[np.ndarray]:
    """
    Extract frames at keyframe_indices and return as list (for later export).
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    frames = []
    indices_set = set(keyframe_indices)
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in indices_set:
                frames.append(frame)
            frame_idx += 1
    finally:
        cap.release()
    return frames


def export_summary_video(
    frames: List[np.ndarray],
    output_path: Union[str, Path],
    fps: int = 15,
    codec: str = "mp4v",
) -> Path:
    """
    Write list of BGR frames to video file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames to export")
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return output_path
