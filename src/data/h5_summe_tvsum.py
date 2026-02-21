"""
Load SumMe/TVSum from eccv16 .h5 format (e.g. eccv16_dataset_summe_google_pool5.h5).

Structure per key: /features (n_steps, dim), /gtscore (n_steps), optional /video_name.
We export labels (gtscore) and features so no .mat or raw videos are needed.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False


def load_h5_dataset(h5_path: Path) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """
    Load one eccv16 .h5 file. Returns list of (video_id, gtscore, features).
    video_id = key or video_name if present.
    """
    if not HAS_H5PY:
        raise ImportError("h5py required for .h5 datasets. Install: pip install h5py")
    h5_path = Path(h5_path)
    if not h5_path.exists():
        return []
    out = []
    with h5py.File(h5_path, "r") as f:
        for key in f.keys():
            try:
                g = f[key]
                if "gtscore" not in g or "features" not in g:
                    continue
                gtscore = np.array(g["gtscore"]).flatten().astype(np.float32)
                features = np.array(g["features"])
                if gtscore.size == 0 or features.size == 0:
                    continue
                # Optional: use human-readable name for SumMe
                video_id = key
                if "video_name" in g:
                    try:
                        vn = np.array(g["video_name"]).flatten()
                        if vn.size > 0:
                            v = vn[0]
                            video_id = v.decode("utf-8") if isinstance(v, bytes) else str(v)
                    except Exception:
                        pass
                # Normalize to [0,1] if needed
                if gtscore.max() > 1 or gtscore.min() < 0:
                    mi, ma = gtscore.min(), gtscore.max()
                    if ma > mi:
                        gtscore = (gtscore - mi) / (ma - mi)
                out.append((video_id, gtscore, features))
            except Exception as e:
                logger.debug("Skip key %s: %s", key, e)
    return out


def prepare_from_h5(
    data_root: Path,
    labels_dir: Path,
    features_dir: Path,
    meta_path: Path | None = None,
) -> int:
    """
    Find eccv16 *_summe_*.h5 and *_tvsum_*.h5 under data_root; export labels and features.
    Writes feature_dim to meta_path (e.g. data/features/_meta.json) for train to use.
    Returns number of videos prepared.
    """
    labels_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    feature_dim = None
    for h5_path in data_root.rglob("*.h5"):
        name = h5_path.name.lower()
        if "summe" not in name and "tvsum" not in name:
            continue
        items = load_h5_dataset(h5_path)
        for video_id, gtscore, features in items:
            if feature_dim is None and features.size > 0:
                feature_dim = features.shape[-1] if features.ndim >= 2 else features.size
            safe_id = video_id.replace("/", "_").replace("\\", "_").strip() or f"video_{count}"
            labels_dir.joinpath(f"{safe_id}.json").write_text(
                json.dumps({"scores": gtscore.tolist()}, indent=None),
                encoding="utf-8",
            )
            np.save(features_dir / f"{safe_id}.npy", features.astype(np.float32))
            count += 1
        logger.info("From %s: %d videos", h5_path.name, len(items))
    if meta_path is not None and feature_dim is not None:
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.write_text(json.dumps({"feature_dim": int(feature_dim)}, indent=None), encoding="utf-8")
        logger.info("Wrote %s (feature_dim=%s) for train/eval", meta_path, feature_dim)
    return count
