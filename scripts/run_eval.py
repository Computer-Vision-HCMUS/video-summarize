#!/usr/bin/env python3
"""CLI: Evaluate model on test set (F-score, temporal overlap)."""

from __future__ import annotations

import argparse
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

    feature_loader = FeatureLoader(
        config.paths.features_root,
        feature_dim=config.features.feature_dim,
    )
    label_loader = LabelLoader(config.paths.labels_root)
    paths = config.paths
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
            scores, _ = model(feats, lengths)
            for i in range(feats.size(0)):
                length = lengths[i].item()
                pred_bin = scores_to_binary(scores[i], length, ratio)
                gt_bin = labels[i][:length].numpy()
                if gt_bin.size < length:
                    gt_bin = np.pad(gt_bin, (0, length - gt_bin.size), constant_values=0)
                p, r, f = compute_precision_recall_fscore(pred_bin, gt_bin)
                all_p.append(p)
                all_r.append(r)
                all_f.append(f)
                keyframes = select_keyshots(scores[i].cpu().numpy(), length, ratio, min_kf)
                pred_ranges = keyshots_to_ranges(keyframes)
                gt_ranges = keyshots_to_ranges(np.where(gt_bin > 0.5)[0].tolist())
                overlap_list.append(temporal_overlap(pred_ranges, gt_ranges, length))

    print(f"Precision: {np.mean(all_p):.4f} | Recall: {np.mean(all_r):.4f} | F-score: {np.mean(all_f):.4f}")
    print(f"Temporal overlap: {np.mean(overlap_list):.4f}")


if __name__ == "__main__":
    main()
