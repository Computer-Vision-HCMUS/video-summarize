#!/usr/bin/env python3
"""CLI: Evaluate model on test set (F-score, temporal overlap)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config.seed)
    setup_logging()

    paths = config.paths

    # Load checkpoint sớm để suy ra đúng input_dim của mô hình
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    # lstm.weight_ih_l0 có shape (4*hidden_size, input_dim)
    lstm_w = state_dict["lstm.weight_ih_l0"]
    model_input_dim = lstm_w.shape[1]

    # Đọc feature_dim từ dữ liệu (hoặc fallback từ config) để biết chiều thực tế của features trên đĩa
    feature_dim = _feature_dim_from_meta(paths.features_root, config.features.feature_dim)

    feature_loader = FeatureLoader(paths.features_root, feature_dim=feature_dim)
    label_loader = LabelLoader(paths.labels_root)
    video_ids = _collect_video_ids(Path(paths.features_root), Path(paths.labels_root))
    if not video_ids:
        raise SystemExit("No videos with features and labels.")

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

    model = BiLSTMSummarizer(
        input_dim=model_input_dim,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        bidirectional=getattr(config.model, "bidirectional", True),
        use_attention=getattr(config.model, "use_attention", True),
    )
    model.load_state_dict(state_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    ratio = config.inference.summary_ratio
    min_kf = config.inference.min_keyframes
    all_p, all_r, all_f = [], [], []
    overlap_list = []

    with torch.no_grad():
        for batch in test_loader:
            feats, labels, lengths, _ = batch
            feats = feats.to(device)
            lengths = lengths.to(device)
            # Nếu checkpoint là multimodal (ví dụ 1408) nhưng features hiện tại chỉ là visual (ví dụ 1024),
            # thì pad thêm các chiều còn thiếu bằng 0 để khớp với input_dim của mô hình.
            if feats.size(-1) != model_input_dim:
                if feats.size(-1) < model_input_dim:
                    pad_dim = model_input_dim - feats.size(-1)
                    pad = torch.zeros(feats.size(0), feats.size(1), pad_dim, device=feats.device)
                    feats = torch.cat([feats, pad], dim=-1)
                else:
                    # Trường hợp hiếm: features có chiều lớn hơn model_input_dim → cắt bớt cho khớp checkpoint
                    feats = feats[..., :model_input_dim]

            scores, _ = model(feats, lengths)
            for i in range(feats.size(0)):
                length = lengths[i].item()
                pred_bin = scores_to_binary(scores[i], length, ratio)
                # GT from .h5 is continuous gtscore; binarize same way (top ratio) for fair P/R/F
                gt_scores = labels[i][:length]
                gt_bin = scores_to_binary(gt_scores, length, ratio)
                p, r, f = compute_precision_recall_fscore(pred_bin, gt_bin)
                all_p.append(p)
                all_r.append(r)
                all_f.append(f)
                keyframes = select_keyshots(scores[i].cpu().numpy(), length, ratio, min_kf)
                pred_ranges = keyshots_to_ranges(keyframes)
                gt_keyframes = np.where(gt_bin > 0.5)[0].tolist()
                gt_ranges = keyshots_to_ranges(gt_keyframes)
                overlap_list.append(temporal_overlap(pred_ranges, gt_ranges, length))

    print(f"Precision: {np.mean(all_p):.4f} | Recall: {np.mean(all_r):.4f} | F-score: {np.mean(all_f):.4f}")
    print(f"Temporal overlap: {np.mean(overlap_list):.4f}")


if __name__ == "__main__":
    main()
