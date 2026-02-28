#!/usr/bin/env python3
"""CLI: Run inference for static/dynamic summary."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.config import load_config
from src.data import FeatureLoader
from src.models import BiLSTMSummarizer
from src.inference import InferencePipeline
from src.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Inference: static/dynamic summary")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint path")
    parser.add_argument("--video-id", type=str, required=True, help="Video ID (stem)")
    parser.add_argument("--video-path", type=str, required=True, help="Path to raw video")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--static", action="store_true", help="Output static keyframes")
    parser.add_argument("--dynamic", action="store_true", help="Output skim video")
    args = parser.parse_args()

    config = load_config(args.config)
    setup_logging()
    output_dir = Path(args.output_dir or config.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feature_loader = FeatureLoader(
        config.paths.features_root,
        feature_dim=config.features.feature_dim,
    )
    model = BiLSTMSummarizer(
        input_dim=config.model.input_dim,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        bidirectional=getattr(config.model, "bidirectional", True),
        use_attention=getattr(config.model, "use_attention", True),
    )
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = InferencePipeline(
        model=model,
        feature_loader=feature_loader,
        device=device,
        summary_ratio=config.inference.summary_ratio,
        min_keyframes=config.inference.min_keyframes,
        skim_fps=config.inference.skim_fps,
        max_seq_len=getattr(config.data, "max_seq_len", 960),
    )

    if args.static:
        keyframes = pipeline.run_static_summary(
            args.video_path, args.video_id,
            output_dir=output_dir / "static" / args.video_id,
        )
        print(f"Static summary: {len(keyframes)} keyframes")
    if args.dynamic:
        out_path = pipeline.run_dynamic_summary(
            args.video_path, args.video_id,
            output_path=output_dir / "dynamic" / f"{args.video_id}_skim.mp4",
        )
        print(f"Dynamic summary: {out_path}")


if __name__ == "__main__":
    main()
