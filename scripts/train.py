#!/usr/bin/env python3
"""CLI: Train BiLSTM video summarization model."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import load_config
from src.data import FeatureLoader, LabelLoader, create_dataloaders
from src.utils import set_seed, setup_logging
from src.training import Trainer


def _collect_video_ids(data_root: Path, features_root: Path, labels_root: Path) -> list[str]:
    """Collect video IDs that have both features and labels."""
    label_files = set(f.stem for f in labels_root.glob("*.json"))
    feature_files = set(f.stem for f in features_root.glob("*.npy"))
    return sorted(label_files & feature_files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BiLSTM summarization model")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Config path")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--log-dir", type=str, default=None, help="Log directory")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)
    log_dir = args.log_dir or config.paths.logs_dir
    setup_logging(log_file=Path(log_dir) / "train.log")

    paths = config.paths
    feature_loader = FeatureLoader(
        paths.features_root,
        feature_dim=config.features.feature_dim,
        padding_value=config.data.padding_value,
    )
    label_loader = LabelLoader(paths.labels_root)
    video_ids = _collect_video_ids(
        Path(paths.data_root),
        Path(paths.features_root),
        Path(paths.labels_root),
    )
    if not video_ids:
        raise SystemExit("No videos with both features and labels found.")

    train_loader, val_loader, _ = create_dataloaders(
        video_ids=video_ids,
        feature_loader=feature_loader,
        label_loader=label_loader,
        train_ratio=config.data.train_split,
        val_ratio=config.data.val_split,
        test_ratio=config.data.test_split,
        batch_size=config.training.batch_size,
        max_seq_len=config.data.max_seq_len,
        min_seq_len=config.data.min_seq_len,
        padding_value=config.data.padding_value,
        seed=config.seed,
    )

    trainer = Trainer(config)
    result = trainer.run(
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=paths.checkpoints_dir,
        resume_path=args.resume,
    )
    print("Best epoch:", result["best_epoch"], "Best val loss:", result["best_val_loss"])


if __name__ == "__main__":
    main()
