"""Keyshot selection from frame importance scores."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def select_keyshots(
    scores: np.ndarray,
    length: int,
    summary_ratio: float,
    min_keyframes: int = 5,
) -> List[int]:
    """
    Select keyframe indices by score. Returns sorted list of frame indices.
    Uses top-k by score; optionally enforce minimum keyframes.
    """
    scores = np.asarray(scores).flatten()[:length]
    k = max(min_keyframes, int(round(length * summary_ratio)))
    k = min(k, length)
    top_indices = np.argsort(scores)[-k:]
    return sorted(top_indices.tolist())


def keyshots_to_ranges(
    keyframe_indices: List[int],
    gap_threshold: int = 30,
) -> List[Tuple[int, int]]:
    """
    Merge nearby keyframes into (start, end) ranges for temporal overlap.
    Consecutive keyframes within gap_threshold are merged into one segment.
    """
    if not keyframe_indices:
        return []
    indices = sorted(keyframe_indices)
    ranges = []
    start = indices[0]
    end = indices[0]
    for i in indices[1:]:
        if i - end <= gap_threshold:
            end = i
        else:
            ranges.append((start, end + 1))
            start = i
            end = i
    ranges.append((start, end + 1))
    return ranges
