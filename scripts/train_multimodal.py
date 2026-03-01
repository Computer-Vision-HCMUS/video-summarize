#!/usr/bin/env python3
"""
CLI: Train BiLSTM với multimodal features (visual + audio).

Thay đổi so với train.py gốc:
  - Dùng MultimodalFeatureLoader thay vì FeatureLoader
  - input_dim = visual_dim + audio_dim (1024+384=1408 hoặc 2048+384=2432)
  - Tự động detect audio coverage và log

Usage:
    python -m scripts.train_multimodal --config configs/config.yaml

Chuẩn bị trước:
    python -m scripts.extract_features --audio   # extract cả visual + audio
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.data import LabelLoader, create_dataloaders
from src.data.multimodal_loader import MultimodalFeatureLoader
from src.utils import set_seed, setup_logging
from src.training import Trainer


AUDIO_DIM = 384  # SentenceBERT all-MiniLM-L6-v2


def _feature_dim_from_meta(features_root: Path, fallback: int) -> int:
    meta = Path(features_root) / "_meta.json"
    if meta.exists():
        try:
            d = json.loads(meta.read_text(encoding="utf-8"))
            return int(d.get("feature_dim", fallback))
        except Exception:
            pass
    return fallback


def _collect_video_ids(features_root: Path, labels_root: Path) -> list[str]:
    label_files   = set(f.stem for f in labels_root.glob("*.json"))
    feature_files = set(f.stem for f in features_root.glob("*.npy")
                        if not f.stem.endswith("_audio"))  # loại audio files
    return sorted(label_files & feature_files)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BiLSTM với Visual + Audio features")
    parser.add_argument("--config",      type=str, default="configs/config.yaml")
    parser.add_argument("--resume",      type=str, default=None)
    parser.add_argument("--audio-dim",   type=int, default=AUDIO_DIM)
    parser.add_argument("--audio-weight", type=float, default=1.0,
                        help="Scale factor cho audio features (default: 1.0)")
    parser.add_argument("--log-dir",     type=str, default=None)
    args = parser.parse_args()

    config  = load_config(args.config)
    set_seed(config.seed)
    log_dir = args.log_dir or config.paths.logs_dir
    setup_logging(log_file=Path(log_dir) / "train_multimodal.log")

    paths       = config.paths
    visual_dim  = _feature_dim_from_meta(
        Path(paths.features_root), config.features.feature_dim
    )
    input_dim   = visual_dim + args.audio_dim   # 1024+384 = 1408

    # Dùng MultimodalFeatureLoader thay vì FeatureLoader
    feature_loader = MultimodalFeatureLoader(
        features_root=paths.features_root,
        visual_dim=visual_dim,
        audio_dim=args.audio_dim,
        audio_weight=args.audio_weight,
        padding_value=config.data.padding_value,
    )

    label_loader = LabelLoader(paths.labels_root)
    video_ids    = _collect_video_ids(
        Path(paths.features_root), Path(paths.labels_root)
    )

    if not video_ids:
        raise SystemExit("Không tìm thấy video nào có cả features và labels.")

    # Log audio coverage
    coverage = feature_loader.audio_coverage(video_ids)
    print(f"[Data] {len(video_ids)} videos | Audio coverage: {coverage:.1%}")
    if coverage < 0.5:
        print("⚠️  Ít hơn 50% videos có audio features.")
        print("   Chạy: python -m scripts.extract_features --audio")

    print(f"[Model] input_dim={input_dim} (visual={visual_dim} + audio={args.audio_dim})")

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
    result  = trainer.run(
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=paths.checkpoints_dir,
        resume_path=args.resume,
        model_input_dim=input_dim,   # override để BiLSTM nhận đúng dim
    )
    print(f"Best epoch: {result['best_epoch']} | Best val loss: {result['best_val_loss']:.4f}")
    print(f"Checkpoint: {paths.checkpoints_dir}/best.pt")


if __name__ == "__main__":
    main()
