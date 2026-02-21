"""
SumMe and TVSum dataset adapter.

Reads .mat annotation files and converts to our label format: JSON with
{"scores": [0.0, 0.2, ...]} (frame-level importance in [0, 1]).

Expected .mat structure (typical):
- TVSum: struct array with 'video' or 'tvsum50', each item has video id and
  frame-level scores (e.g. user_anno [n_frames x n_users], we average).
- SumMe: struct with video names and annotations (e.g. multiple keyframe sets),
  we convert to frame-level scores (proportion of users that selected frame).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from scipy.io import loadmat
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _safe_loadmat(path: Path) -> Optional[Any]:
    if not HAS_SCIPY:
        raise ImportError("scipy is required for SumMe/TVSum. Install with: pip install scipy")
    if not path.exists():
        return None
    try:
        return loadmat(str(path), struct_as_record=False, squeeze_me=True)
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return None


def _normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Map scores to [0, 1]."""
    scores = np.asarray(scores, dtype=np.float32)
    scores = np.nan_to_num(scores, nan=0.0)
    if scores.size == 0:
        return scores
    min_s, max_s = scores.min(), scores.max()
    if max_s > min_s:
        scores = (scores - min_s) / (max_s - min_s)
    return scores


def load_tvsum_mat(mat_path: str | Path) -> List[Tuple[str, np.ndarray]]:
    """
    Load TVSum .mat file. Returns list of (video_id, scores) where scores is (n_frames,) in [0,1].

    Handles common layouts:
    - 'dataset' or 'tvsum50': array of structs with 'video', 'nframes', 'user_anno' or 'importance'
    - 'video' / 'video_name' and frame-level annotations
    """
    data = _safe_loadmat(Path(mat_path))
    if data is None:
        return []

    out: List[Tuple[str, np.ndarray]] = []
    # Common key names in TVSum releases
    for key in ("dataset", "tvsum50", "tvsum", "data"):
        if key not in data:
            continue
        raw = data[key]
        if not hasattr(raw, "__len__"):
            raw = [raw]
        for item in np.atleast_1d(raw):
            try:
                vid = getattr(item, "video", None) or getattr(item, "video_name", None) or getattr(item, "title", None)
                if vid is None:
                    continue
                vid = str(vid).strip()
                # Frame-level importance: (n_frames,) or (n_frames, n_users)
                imp = getattr(item, "user_anno", None) or getattr(item, "importance", None) or getattr(item, "anno", None)
                if imp is None:
                    nf = getattr(item, "nframes", None) or getattr(item, "n_frames", None)
                    if nf is not None:
                        imp = np.zeros(int(nf), dtype=np.float32)
                imp = np.asarray(imp)
                if imp.ndim == 2:
                    imp = np.mean(imp, axis=1)
                imp = _normalize_scores(imp)
                out.append((vid, imp))
            except Exception as e:
                logger.debug("Skip item in TVSum .mat: %s", e)
        if out:
            break

    return out


def load_summe_mat(mat_path: str | Path) -> List[Tuple[str, np.ndarray]]:
    """
    Load SumMe .mat file. Returns list of (video_id, scores).

    SumMe annotations are often multiple user summaries (keyframe indices or segments).
    We build frame-level scores: for each frame, score = proportion of users that selected it.
    """
    data = _safe_loadmat(Path(mat_path))
    if data is None:
        return []

    out: List[Tuple[str, np.ndarray]] = []
    # Common keys
    for key in ("video", "videos", "video_name", "names"):
        if key not in data:
            continue
        names_raw = data[key]
        ann_key = "anno" if "anno" in data else "annotations"
        ann_raw = data.get(ann_key, None)
        if ann_raw is None:
            continue
        names = np.atleast_1d(names_raw)
        if hasattr(names[0], "strip"):
            names = [str(n).strip() for n in names]
        else:
            names = [str(n) for n in names]
        anns = np.atleast_1d(ann_raw)
        for i, vid in enumerate(names):
            if i >= len(anns):
                break
            try:
                ann = anns[i]
                # ann can be (n_frames, n_users) 0/1 or list of keyframe indices per user
                ann = np.asarray(ann)
                if ann.ndim == 1:
                    # One vector of frame scores or one set of keyframe indices
                    scores = ann.astype(np.float32)
                    if scores.max() <= 1 and scores.min() >= 0:
                        pass
                    else:
                        # Treat as keyframe indices
                        L = int(ann.max()) + 1 if ann.size else 0
                        scores = np.zeros(L, dtype=np.float32)
                        for idx in np.atleast_1d(ann):
                            if 0 <= int(idx) < L:
                                scores[int(idx)] = 1.0
                elif ann.ndim == 2:
                    scores = np.mean(ann.astype(np.float32), axis=1)
                else:
                    continue
                scores = _normalize_scores(scores)
                out.append((vid, scores))
            except Exception as e:
                logger.debug("Skip SumMe item %s: %s", vid, e)
        if out:
            break

    return out


def export_labels_to_json(
    label_pairs: List[Tuple[str, np.ndarray]],
    output_dir: str | Path,
) -> List[Path]:
    """Write each (video_id, scores) to output_dir/{video_id}.json."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for vid, scores in label_pairs:
        path = output_dir / f"{vid}.json"
        obj = {"scores": scores.tolist()}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f)
        written.append(path)
    return written
