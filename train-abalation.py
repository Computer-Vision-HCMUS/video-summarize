#!/usr/bin/env python3
"""Train ablation variants and export checkpoint .pt files."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from src.config import Config, load_config
from src.data import FeatureLoader, LabelLoader, create_dataloaders
from src.data.multimodal_loader import MultimodalFeatureLoader
from src.training import Trainer
from src.utils import set_seed, setup_logging

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _feature_dim_from_meta(features_root: Path, fallback: int) -> int:
    meta = features_root / "_meta.json"
    if meta.exists():
        try:
            data = json.loads(meta.read_text(encoding="utf-8"))
            return int(data.get("feature_dim", fallback))
        except Exception:
            pass
    return fallback


def _collect_video_ids(features_root: Path, labels_root: Path) -> list[str]:
    label_files = {f.stem for f in labels_root.glob("*.json")}
    feature_files = {
        f.stem
        for f in features_root.glob("*.npy")
        if not f.stem.endswith("_audio")
    }
    return sorted(label_files & feature_files)


def _clone_config_with_overrides(
    base_config,
    *,
    use_attention: bool,
    bidirectional: bool,
    input_dim: int,
) -> Config:
    data: dict[str, Any] = base_config.to_dict()
    model_data = dict(data.get("model", {}))
    model_data["use_attention"] = bool(use_attention)
    model_data["bidirectional"] = bool(bidirectional)
    model_data["input_dim"] = int(input_dim)
    data["model"] = model_data
    return Config(data)


def _make_loader(config, variant: str, visual_dim: int):
    features_root = Path(config.paths.features_root)
    if variant == "visual_only":
        return FeatureLoader(
            features_root,
            feature_dim=visual_dim,
            padding_value=config.data.padding_value,
        ), visual_dim

    audio_dim = int(getattr(config.model, "audio_dim", 384))
    return MultimodalFeatureLoader(
        features_root=features_root,
        visual_dim=visual_dim,
        audio_dim=audio_dim,
        padding_value=config.data.padding_value,
    ), visual_dim + audio_dim


def _train_variant(
    *,
    base_config,
    variant_key: str,
    variant_label: str,
    use_attention: bool,
    bidirectional: bool,
    out_checkpoint_path: Path,
) -> None:
    paths = base_config.paths
    features_root = Path(paths.features_root)
    labels_root = Path(paths.labels_root)
    checkpoints_root = Path(paths.checkpoints_dir)
    logs_root = Path(paths.logs_dir)

    visual_dim = _feature_dim_from_meta(features_root, base_config.features.feature_dim)
    feature_loader, model_input_dim = _make_loader(base_config, variant_key, visual_dim)
    label_loader = LabelLoader(labels_root)
    video_ids = _collect_video_ids(features_root, labels_root)
    if not video_ids:
        raise SystemExit("No videos with both labels and features found.")

    variant_config = _clone_config_with_overrides(
        base_config,
        use_attention=use_attention,
        bidirectional=bidirectional,
        input_dim=model_input_dim,
    )

    train_loader, val_loader, _ = create_dataloaders(
        video_ids=video_ids,
        feature_loader=feature_loader,
        label_loader=label_loader,
        train_ratio=variant_config.data.train_split,
        val_ratio=variant_config.data.val_split,
        test_ratio=variant_config.data.test_split,
        batch_size=variant_config.training.batch_size,
        max_seq_len=variant_config.data.max_seq_len,
        min_seq_len=variant_config.data.min_seq_len,
        padding_value=variant_config.data.padding_value,
        seed=variant_config.seed,
    )

    run_ckpt_dir = checkpoints_root / "ablation_runs" / variant_key
    run_log_path = logs_root / f"train_ablation_{variant_key}.log"
    run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_log_path.parent.mkdir(parents=True, exist_ok=True)
    setup_logging(log_file=run_log_path)

    trainer = Trainer(variant_config)
    print(f"\n[train] {variant_label}")
    print(
        f"        input_dim={model_input_dim} | "
        f"use_attention={use_attention} | bidirectional={bidirectional}"
    )
    result = trainer.run(
        train_loader=train_loader,
        val_loader=val_loader,
        checkpoint_dir=run_ckpt_dir,
        resume_path=None,
        model_input_dim=model_input_dim,
    )

    best_path = run_ckpt_dir / "best.pt"
    if not best_path.exists():
        raise RuntimeError(f"Best checkpoint missing for variant '{variant_key}': {best_path}")

    out_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, out_checkpoint_path)
    print(
        f"[ok] {variant_label}: best_epoch={result['best_epoch']} "
        f"best_val_loss={result['best_val_loss']:.4f}"
    )
    print(f"     exported -> {out_checkpoint_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train ablation variants and export checkpoints for abalation.py --mode variants"
    )
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("visual_only", "no_attention", "unilstm", "full"),
        default=("visual_only", "no_attention", "unilstm"),
        help="Which variants to train.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="checkpoints",
        help="Output directory for exported ablation *.pt files.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)

    out_dir = Path(args.out_dir)
    variants = {
        "full": {
            "label": "Full model (multimodal + attention + BiLSTM)",
            "use_attention": True,
            "bidirectional": True,
            "out_name": "ablation_full.pt",
        },
        "visual_only": {
            "label": "Ablation visual-only",
            "use_attention": True,
            "bidirectional": True,
            "out_name": "ablation_visual_only.pt",
        },
        "no_attention": {
            "label": "Ablation no-attention",
            "use_attention": False,
            "bidirectional": True,
            "out_name": "ablation_no_attention.pt",
        },
        "unilstm": {
            "label": "Ablation UniLSTM",
            "use_attention": True,
            "bidirectional": False,
            "out_name": "ablation_unilstm.pt",
        },
    }

    for key in args.variants:
        cfg = variants[key]
        _train_variant(
            base_config=config,
            variant_key=key,
            variant_label=cfg["label"],
            use_attention=cfg["use_attention"],
            bidirectional=cfg["bidirectional"],
            out_checkpoint_path=out_dir / cfg["out_name"],
        )

    print("\nDone.")
    print("Use these checkpoints with:")
    print(
        "python abalation.py --mode variants --full checkpoints/best.pt "
        "--visual-only checkpoints/ablation_visual_only.pt "
        "--no-attention checkpoints/ablation_no_attention.pt "
        "--unilstm checkpoints/ablation_unilstm.pt --output abalation.md"
    )


if __name__ == "__main__":
    main()
