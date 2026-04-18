"""
Keyshot selection — hybrid strategy: quality + coverage.

Fix 2: Thay diversity cứng (chia đều k segments) bằng hybrid approach:
  - 70% keyframes: top-k theo score → giữ key points quan trọng nhất
  - 30% keyframes: diversity → đảm bảo cover đủ các phần video

Tại sao hybrid tốt hơn:
  - Diversity cứng 100%: bắt buộc chọn từ mọi đoạn kể cả score thấp → miss key points
  - Top-k 100%: dồn vào 1 đoạn, bỏ sót nội dung phần khác
  - Hybrid 70/30: giữ được key points + đảm bảo coverage cơ bản
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def smooth_scores(scores: np.ndarray, sigma: float = 3.0) -> np.ndarray:
    """Gaussian smoothing để loại spike nhiễu."""
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) < 5:
        return scores
    return gaussian_filter1d(scores, sigma=sigma)


def detect_shots(scores: np.ndarray, min_shot_len: int = 4) -> List[Tuple[int, int]]:
    """Chia sequence thành shots có độ dài cố định."""
    T = len(scores)
    shots, start = [], 0
    while start < T:
        end = min(start + min_shot_len, T)
        shots.append((start, end))
        start = end
    return shots


def _best_frame_in_shot(
    shot: Tuple[int, int], smoothed: np.ndarray
) -> int:
    """Frame đại diện = frame có score cao nhất trong shot."""
    start, end = shot
    return start + int(np.argmax(smoothed[start:end]))


def _enforce_min_gap(keyframes: List[int], min_gap: int) -> List[int]:
    """Loại bỏ keyframes quá gần nhau."""
    if not keyframes:
        return []
    result = [keyframes[0]]
    for kf in keyframes[1:]:
        if kf - result[-1] >= min_gap:
            result.append(kf)
    return result


# ─────────────────────────────────────────────────────────────
# Core: Hybrid selection
# ─────────────────────────────────────────────────────────────

def select_keyshots_improved(
    scores: np.ndarray,
    length: int,
    summary_ratio: float = 0.15,
    min_keyframes: int = 3,
    sigma: float = 3.0,
    min_gap: int = 8,
    shot_len: int = 6,
    quality_ratio: float = 0.7,   # 70% top-k quality, 30% diversity
) -> List[int]:
    """
    Hybrid keyshot selection: quality-first + diversity fallback.

    Args:
        scores:        [T] raw importance scores từ model.
        length:        Actual video length (ignore padding).
        summary_ratio: Tỷ lệ tóm tắt (0.15 = 15% video).
        min_keyframes: Số keyframes tối thiểu.
        sigma:         Gaussian smoothing sigma.
        min_gap:       Khoảng cách tối thiểu giữa 2 keyframes (frames).
        shot_len:      Độ dài mỗi shot để group frames.
        quality_ratio: Tỷ lệ keyframes từ top-k (còn lại từ diversity).

    Returns:
        Sorted list of keyframe indices.
    """
    scores   = np.asarray(scores).flatten()[:length]
    T        = len(scores)

    if T < min_keyframes:
        return list(range(T))

    # 1. Smooth
    smoothed = smooth_scores(scores, sigma=sigma)

    # 2. Shots
    shots       = detect_shots(smoothed, min_shot_len=shot_len)
    n_shots     = len(shots)
    shot_scores = np.array([float(np.max(smoothed[s:e])) for s, e in shots])

    # 3. Tổng số keyframes cần chọn
    k_total   = max(min_keyframes, int(round(T * summary_ratio / shot_len)))
    k_total   = min(k_total, n_shots)

    # 4. Phân chia quality vs diversity
    k_quality   = max(1, int(round(k_total * quality_ratio)))
    k_diversity = max(0, k_total - k_quality)

    # 5. Quality: top-k shots theo score
    top_shot_ids = set(np.argsort(shot_scores)[-k_quality:])
    quality_kfs  = [_best_frame_in_shot(shots[sid], smoothed) for sid in top_shot_ids]

    # 6. Diversity: chia video thành k_diversity segments, chọn best shot
    #    từ mỗi segment mà CHƯA được chọn bởi quality
    diversity_kfs = []
    if k_diversity > 0:
        remaining_ids = [i for i in range(n_shots) if i not in top_shot_ids]
        if remaining_ids:
            segments = np.array_split(remaining_ids, k_diversity)
            for seg in segments:
                if len(seg) == 0:
                    continue
                seg_scores = shot_scores[seg]
                best       = seg[int(np.argmax(seg_scores))]
                diversity_kfs.append(_best_frame_in_shot(shots[best], smoothed))

    # 7. Merge, sort, enforce gap
    all_kfs = sorted(set(quality_kfs + diversity_kfs))
    all_kfs = _enforce_min_gap(all_kfs, min_gap)

    return sorted(all_kfs)


# ─────────────────────────────────────────────────────────────
# Backward compatible
# ─────────────────────────────────────────────────────────────

def select_keyshots(
    scores: np.ndarray,
    length: int,
    summary_ratio: float = 0.15,
    min_keyframes: int = 5,
) -> List[int]:
    return select_keyshots_improved(
        scores=scores,
        length=length,
        summary_ratio=summary_ratio,
        min_keyframes=min_keyframes,
    )


def keyshots_to_ranges(
    keyframe_indices: List[int],
    gap_threshold: int = 30,
) -> List[Tuple[int, int]]:
    if not keyframe_indices:
        return []
    indices = sorted(keyframe_indices)
    ranges, start, end = [], indices[0], indices[0]
    for i in indices[1:]:
        if i - end <= gap_threshold:
            end = i
        else:
            ranges.append((start, end + 1))
            start = i
            end   = i
    ranges.append((start, end + 1))
    return ranges
