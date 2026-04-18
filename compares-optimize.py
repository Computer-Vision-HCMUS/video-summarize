#!/usr/bin/env python3
"""
So sánh độ chính xác trên tập test giữa checkpoint baseline và checkpoint optimize.

Chạy từ thư mục gốc project `video-summarize` (hoặc bất kỳ đâu — script tự thêm path):

  python compares-optimize.py
  python compares-optimize.py --baseline checkpoints/best.pt --optimize checkpoints-optimize/best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.data import FeatureLoader, LabelLoader, create_dataloaders
from src.models import BiLSTMSummarizer
from src.evaluation import (
    compute_precision_recall_fscore,
    temporal_overlap,
    scores_to_binary,
    select_keyshots,
    keyshots_to_ranges,
)
from src.utils import set_seed, setup_logging


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
    label_files = set(f.stem for f in labels_root.glob("*.json"))
    feature_files = set(f.stem for f in features_root.glob("*.npy"))
    return sorted(label_files & feature_files)


def _checkpoint_meta(ckpt: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in ("epoch", "best_val_loss"):
        if k in ckpt:
            out[k] = ckpt[k]
    return out


def _input_dim_from_state(state_dict: dict[str, Any]) -> int:
    w = state_dict["lstm.weight_ih_l0"]
    return int(w.shape[1])


def evaluate_checkpoint(
    checkpoint_path: Path,
    config,
    test_loader,
    device: torch.device,
) -> dict[str, float]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    model_input_dim = _input_dim_from_state(state_dict)
    bidirectional = getattr(config.model, "bidirectional", True)
    use_attention, fuse_attention = BiLSTMSummarizer.infer_attention_flags_from_state_dict(
        state_dict,
        hidden_size=config.model.hidden_size,
        bidirectional=bidirectional,
    )

    model = BiLSTMSummarizer(
        input_dim=model_input_dim,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        bidirectional=bidirectional,
        use_attention=use_attention,
        fuse_attention_context=fuse_attention,
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()
    print(
        f"  Đã load {checkpoint_path.name}: "
        f"use_attention={use_attention}, fuse_attention_context={fuse_attention}"
    )

    ratio = config.inference.summary_ratio
    min_kf = config.inference.min_keyframes
    all_p, all_r, all_f = [], [], []
    overlap_list = []

    with torch.no_grad():
        for batch in test_loader:
            feats, labels, lengths, _ = batch
            feats = feats.to(device)
            lengths = lengths.to(device)

            if feats.size(-1) != model_input_dim:
                if feats.size(-1) < model_input_dim:
                    pad_dim = model_input_dim - feats.size(-1)
                    pad = torch.zeros(
                        feats.size(0), feats.size(1), pad_dim, device=feats.device
                    )
                    feats = torch.cat([feats, pad], dim=-1)
                else:
                    feats = feats[..., :model_input_dim]

            scores, _ = model(feats, lengths)
            for i in range(feats.size(0)):
                length = int(lengths[i].item())
                pred_bin = scores_to_binary(scores[i], length, ratio)
                gt_scores = labels[i][:length]
                gt_bin = scores_to_binary(gt_scores, length, ratio)
                p, r, f = compute_precision_recall_fscore(pred_bin, gt_bin)
                all_p.append(p)
                all_r.append(r)
                all_f.append(f)
                keyframes = select_keyshots(
                    scores[i].cpu().numpy(), length, ratio, min_kf
                )
                pred_ranges = keyshots_to_ranges(keyframes)
                gt_keyframes = np.where(gt_bin > 0.5)[0].tolist()
                gt_ranges = keyshots_to_ranges(gt_keyframes)
                overlap_list.append(
                    temporal_overlap(pred_ranges, gt_ranges, length)
                )

    n = len(all_f)
    if n == 0:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "fscore": 0.0,
            "temporal_overlap": 0.0,
        }

    return {
        "precision": float(np.mean(all_p)),
        "recall": float(np.mean(all_r)),
        "fscore": float(np.mean(all_f)),
        "temporal_overlap": float(np.mean(overlap_list)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="So sánh metrics test: checkpoint baseline vs checkpoint optimize."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="File config (paths data, inference.summary_ratio, ...).",
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default="checkpoints/best.pt",
        help="Checkpoint mô hình baseline (vd. nhánh main / trước optimize).",
    )
    parser.add_argument(
        "--optimize",
        type=str,
        default="checkpoints-optimize/best.pt",
        help="Checkpoint sau khi optimize (thư mục checkpoints-optimize).",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"Không tìm thấy config: {config_path.resolve()}")

    baseline_path = Path(args.baseline)
    optimize_path = Path(args.optimize)
    for p, name in (
        (baseline_path, "baseline"),
        (optimize_path, "optimize"),
    ):
        if not p.is_file():
            raise SystemExit(f"Không tìm thấy checkpoint {name}: {p.resolve()}")

    config = load_config(str(config_path))
    set_seed(config.seed)
    setup_logging()

    paths = config.paths
    feature_dim = _feature_dim_from_meta(
        Path(paths.features_root), config.features.feature_dim
    )
    feature_loader = FeatureLoader(paths.features_root, feature_dim=feature_dim)
    label_loader = LabelLoader(paths.labels_root)
    video_ids = _collect_video_ids(
        Path(paths.features_root), Path(paths.labels_root)
    )
    if not video_ids:
        raise SystemExit("Không có video nào có cả feature và label.")

    _, _, test_loader = create_dataloaders(
        video_ids=video_ids,
        feature_loader=feature_loader,
        label_loader=label_loader,
        train_ratio=config.data.train_split,
        val_ratio=config.data.val_split,
        test_ratio=config.data.test_split,
        batch_size=config.training.batch_size,
        max_seq_len=config.data.max_seq_len,
        min_seq_len=config.data.min_seq_len,
        seed=config.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Test split — cùng seed={config.seed}, ratio inference={config.inference.summary_ratio}\n")

    ckpt_b = torch.load(baseline_path, map_location="cpu")
    ckpt_o = torch.load(optimize_path, map_location="cpu")
    sd_b = ckpt_b.get("model_state_dict", ckpt_b)
    sd_o = ckpt_o.get("model_state_dict", ckpt_o)
    dim_b = _input_dim_from_state(sd_b)
    dim_o = _input_dim_from_state(sd_o)

    print("--- Metadata checkpoint (nếu có khi train) ---")
    print(f"  Baseline:  {baseline_path}  | LSTM input_dim={dim_b}  | {_checkpoint_meta(ckpt_b)}")
    print(f"  Optimize:  {optimize_path}  | LSTM input_dim={dim_o}  | {_checkpoint_meta(ckpt_o)}")
    print()

    print("Đang eval baseline...")
    m_b = evaluate_checkpoint(baseline_path, config, test_loader, device)
    print("Đang eval optimize...")
    m_o = evaluate_checkpoint(optimize_path, config, test_loader, device)

    print()
    print("=" * 72)
    print(f"{'Metric':<22} {'Baseline':>14} {'Optimize':>14} {'Δ (O - B)':>14}")
    print("=" * 72)
    for key, label in (
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("fscore", "F-score"),
        ("temporal_overlap", "Temporal overlap"),
    ):
        b, o = m_b[key], m_o[key]
        d = o - b
        print(f"{label:<22} {b:>14.4f} {o:>14.4f} {d:>+14.4f}")
    print("=" * 72)
    print()
    print("Ghi chú: Δ dương = optimize cao hơn baseline trên metric đó.")


if __name__ == "__main__":
    main()
