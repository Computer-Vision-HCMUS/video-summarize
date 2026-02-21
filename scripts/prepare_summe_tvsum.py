#!/usr/bin/env python3
"""
Prepare SumMe or TVSum dataset for training.

Reads .mat annotation file and writes frame-level labels to data/labels/*.json.
Optionally copies or links videos to data/raw so video_id matches.

Expected layout:
  --dataset-dir data/summe   (or data/tvsum)
    - SumMe: videos in .mp4/.avi, annotations in GT/SumMe.mat or similar
    - TVSum: videos in .mp4/.avi, annotations in gt/tvsum50.mat or similar

Usage:
  python -m scripts.prepare_summe_tvsum --dataset summe --mat path/to/SumMe.mat --labels-dir data/labels
  python -m scripts.prepare_summe_tvsum --dataset tvsum --mat path/to/tvsum50.mat --labels-dir data/labels
"""

from __future__ import annotations

import argparse
from pathlib import Path

from src.data.summe_tvsum import (
    load_tvsum_mat,
    load_summe_mat,
    export_labels_to_json,
)
from src.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare SumMe or TVSum labels")
    parser.add_argument("--dataset", choices=("summe", "tvsum"), required=True)
    parser.add_argument("--mat", type=str, required=True, help="Path to .mat annotation file")
    parser.add_argument("--labels-dir", type=str, default="data/labels", help="Output directory for JSON labels")
    parser.add_argument("--video-dir", type=str, default=None, help="Optional: copy/link videos here (by video_id)")
    args = parser.parse_args()

    setup_logging()
    mat_path = Path(args.mat)
    if not mat_path.exists():
        raise SystemExit(f".mat file not found: {mat_path}")

    if args.dataset == "summe":
        pairs = load_summe_mat(mat_path)
    else:
        pairs = load_tvsum_mat(mat_path)

    if not pairs:
        raise SystemExit("No (video_id, scores) loaded from .mat. Check file structure and key names.")

    written = export_labels_to_json(pairs, args.labels_dir)
    print(f"Exported {len(written)} label files to {args.labels_dir}")
    print("Video IDs (first 5):", [p[0] for p in pairs[:5]])


if __name__ == "__main__":
    main()
