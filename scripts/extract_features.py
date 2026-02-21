#!/usr/bin/env python3
"""CLI: Extract CNN features from raw videos."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.features import CNNFeatureExtractor, FeatureExtractionPipeline, FrameSampler
from src.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract video features")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--video-dir", type=str, default=None, help="Override data_root")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing features")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging()
    video_dir = Path(args.video_dir or config.paths.data_root)
    if not video_dir.exists():
        raise SystemExit(f"Video directory not found: {video_dir}")

    sampler = FrameSampler(sample_rate=config.features.sample_rate)
    extractor = CNNFeatureExtractor(
        backbone=config.features.backbone,
        pretrained=config.features.pretrained,
        feature_dim=config.features.feature_dim,
        device=config.features.device,
    )
    pipeline = FeatureExtractionPipeline(
        output_dir=config.paths.features_root,
        frame_sampler=sampler,
        extractor=extractor,
    )

    video_paths = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi")) + list(video_dir.glob("*.mkv"))
    video_ids = [p.stem for p in video_paths]
    saved = pipeline.process_videos(video_paths, video_ids, overwrite=args.overwrite)
    print(f"Extracted features for {len(saved)} videos.")


if __name__ == "__main__":
    main()
