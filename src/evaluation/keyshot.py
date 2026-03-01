"""
Keyshot selection — cải thiện so với version cũ.

Vấn đề của version cũ:
  - select_keyshots() chọn top-k frame rời rạc → vô nghĩa
  - Không có smoothing → chọn spike đơn lẻ ngẫu nhiên
  - Không có shot awareness → cắt giữa chừng

Cải tiến:
  1. Gaussian smoothing trên scores trước khi chọn
     → tránh chọn spike nhiễu đơn lẻ
  2. Shot-based selection — gom frames thành shots, score cả shot
     → chọn đoạn liên tục có nghĩa thay vì frame lẻ
  3. Minimum gap giữa các keyframes
     → tránh chọn nhiều frames cùng 1 chỗ
  4. Diversity enforcement
     → đảm bảo keyframes trải đều video, không dồn vào 1 đoạn
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

def smooth_scores(
    scores: np.ndarray,
    sigma: float = 3.0,
) -> np.ndarray:
    """
    Gaussian smoothing trên score sequence.
    Loại bỏ spike nhiễu đơn lẻ, giữ lại peaks thực sự quan trọng.

    Args:
        scores: [T] importance scores.
        sigma:  Độ rộng Gaussian (frames). Lớn hơn = smooth hơn.
                sigma=3 tương đương ~1.5 giây với sample_rate=2fps.
    """
    scores = np.asarray(scores, dtype=np.float32)
    if len(scores) < 5:
        return scores
    return gaussian_filter1d(scores, sigma=sigma)


def detect_shots(
    scores: np.ndarray,
    min_shot_len: int = 4,
) -> List[Tuple[int, int]]:
    """
    Chia sequence thành shots dựa trên local minima của scores.
    Mỗi shot là 1 đoạn liên tục (start, end).

    Đơn giản hóa: chia đều thành shots có độ dài cố định.
    Phù hợp khi không có scene detection thực sự.

    Args:
        scores:       [T] smoothed scores.
        min_shot_len: Độ dài tối thiểu mỗi shot (frames).

    Returns:
        List of (start, end) frame indices.
    """
    T = len(scores)
    shots = []
    start = 0
    while start < T:
        end = min(start + min_shot_len, T)
        shots.append((start, end))
        start = end
    return shots


def select_keyshots_improved(
    scores: np.ndarray,
    length: int,
    summary_ratio: float = 0.15,
    min_keyframes: int = 3,
    sigma: float = 3.0,
    min_gap: int = 8,
    shot_len: int = 6,
    diversity: bool = True,
) -> List[int]:
    """
    Cải tiến select_keyshots() — shot-based + smoothing + diversity.

    Flow:
        raw scores → Gaussian smooth → shot scoring → top-k shots
        → representative frame per shot → diversity filter → keyframes

    Args:
        scores:        [T] raw importance scores từ model.
        length:        Actual video length (ignore padding).
        summary_ratio: Tỷ lệ tóm tắt (0.15 = 15% video).
        min_keyframes: Số keyframes tối thiểu.
        sigma:         Gaussian smoothing sigma.
        min_gap:       Khoảng cách tối thiểu giữa 2 keyframes (frames).
        shot_len:      Độ dài mỗi shot để group frames (frames).
        diversity:     Enforce trải đều keyframes trên video.

    Returns:
        Sorted list of keyframe indices.
    """
    scores = np.asarray(scores).flatten()[:length]
    T      = len(scores)

    if T < min_keyframes:
        return list(range(T))

    # 1. Smooth scores
    smoothed = smooth_scores(scores, sigma=sigma)

    # 2. Shot-based scoring
    shots    = detect_shots(smoothed, min_shot_len=shot_len)
    n_shots  = len(shots)

    # Score mỗi shot = max score trong shot (max vì ta muốn đoạn có peak cao)
    shot_scores = []
    for start, end in shots:
        shot_score = float(np.max(smoothed[start:end]))
        shot_scores.append(shot_score)
    shot_scores = np.array(shot_scores)

    # 3. Chọn k shots tốt nhất
    k = max(min_keyframes, int(round(T * summary_ratio / shot_len)))
    k = min(k, n_shots)

    if diversity and k > 1:
        # Diversity: chia video thành k segments, chọn best shot từ mỗi segment
        keyframes = _diversity_select(shots, shot_scores, k, smoothed)
    else:
        # Greedy: top-k shots theo score
        top_shot_ids = np.argsort(shot_scores)[-k:]
        keyframes    = []
        for sid in top_shot_ids:
            start, end = shots[sid]
            # Đại diện shot = frame có score cao nhất trong shot
            rep = start + int(np.argmax(smoothed[start:end]))
            keyframes.append(rep)

    # 4. Enforce minimum gap giữa keyframes
    keyframes = _enforce_min_gap(sorted(keyframes), min_gap)

    return sorted(keyframes)


def _diversity_select(
    shots: List[Tuple[int, int]],
    shot_scores: np.ndarray,
    k: int,
    smoothed: np.ndarray,
) -> List[int]:
    """
    Chia video thành k phần đều nhau, chọn shot tốt nhất từ mỗi phần.
    → Đảm bảo keyframes trải đều trên toàn bộ video.
    """
    n_shots  = len(shots)
    segments = np.array_split(np.arange(n_shots), k)
    keyframes = []
    for seg in segments:
        if len(seg) == 0:
            continue
        seg_scores = shot_scores[seg]
        best_shot  = seg[int(np.argmax(seg_scores))]
        start, end = shots[best_shot]
        rep        = start + int(np.argmax(smoothed[start:end]))
        keyframes.append(rep)
    return keyframes


def _enforce_min_gap(keyframes: List[int], min_gap: int) -> List[int]:
    """
    Loại bỏ keyframes quá gần nhau (< min_gap frames).
    Giữ lại frame đầu tiên của mỗi nhóm.
    """
    if not keyframes:
        return []
    result = [keyframes[0]]
    for kf in keyframes[1:]:
        if kf - result[-1] >= min_gap:
            result.append(kf)
    return result


# ─────────────────────────────────────────────────────────────
# Backward compatible — giữ nguyên API cũ
# ─────────────────────────────────────────────────────────────

def select_keyshots(
    scores: np.ndarray,
    length: int,
    summary_ratio: float = 0.15,
    min_keyframes: int = 5,
) -> List[int]:
    """
    Drop-in replacement cho select_keyshots() cũ.
    Gọi select_keyshots_improved() bên dưới.
    """
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
    """Giữ nguyên từ code cũ — dùng cho evaluation."""
    if not keyframe_indices:
        return []
    indices = sorted(keyframe_indices)
    ranges  = []
    start   = indices[0]
    end     = indices[0]
    for i in indices[1:]:
        if i - end <= gap_threshold:
            end = i
        else:
            ranges.append((start, end + 1))
            start = i
            end   = i
    ranges.append((start, end + 1))
    return ranges