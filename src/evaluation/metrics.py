"""Precision, Recall, F-score and temporal overlap for video summarization."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def compute_precision_recall_fscore(
    pred_binary: np.ndarray,
    gt_binary: np.ndarray,
) -> Tuple[float, float, float]:
    """
    pred_binary, gt_binary: (T,) binary 0/1.
    Returns (precision, recall, f_score).
    """
    pred_binary = np.asarray(pred_binary).flatten()
    gt_binary = np.asarray(gt_binary).flatten()
    T = max(len(pred_binary), len(gt_binary))
    if T == 0:
        return 0.0, 0.0, 0.0
    pred_binary = np.resize(pred_binary, T)
    gt_binary = np.resize(gt_binary, T)

    tp = np.logical_and(pred_binary == 1, gt_binary == 1).sum()
    pred_pos = pred_binary.sum()
    gt_pos = gt_binary.sum()

    precision = tp / pred_pos if pred_pos > 0 else 0.0
    recall = tp / gt_pos if gt_pos > 0 else 0.0
    if precision + recall > 0:
        f_score = 2 * precision * recall / (precision + recall)
    else:
        f_score = 0.0
    return float(precision), float(recall), float(f_score)


def temporal_overlap(
    pred_ranges: list[Tuple[int, int]],
    gt_ranges: list[Tuple[int, int]],
    total_frames: int,
) -> float:
    """
    Compute overlap ratio: sum of intersection lengths / sum of union lengths
    over all segment pairs (or frame-level IoU). Simplified: frame-level overlap.
    pred_ranges, gt_ranges: list of (start, end) frame indices.
    """
    pred_mask = np.zeros(total_frames, dtype=bool)
    gt_mask = np.zeros(total_frames, dtype=bool)
    for (a, b) in pred_ranges:
        pred_mask[max(0, a) : min(b, total_frames)] = True
    for (a, b) in gt_ranges:
        gt_mask[max(0, a) : min(b, total_frames)] = True
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return float(inter / union) if union > 0 else 0.0


def scores_to_binary(
    scores: torch.Tensor | np.ndarray,
    length: int,
    ratio: float,
) -> np.ndarray:
    """Turn continuous scores into binary by top-k ratio."""
    if torch.is_tensor(scores):
        scores = scores.detach().cpu().numpy()
    scores = np.asarray(scores).flatten()[:length]
    k = max(1, int(round(length * ratio)))
    idx = np.argsort(scores)[-k:]
    out = np.zeros(length, dtype=np.float32)
    out[idx] = 1.0
    return out
