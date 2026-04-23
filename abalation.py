#!/usr/bin/env python3
"""Run ablation evaluation and write abalation.md."""

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
from src.data.multimodal_loader import MultimodalFeatureLoader
from src.evaluation import (
    compute_precision_recall_fscore,
    keyshots_to_ranges,
    scores_to_binary,
    select_keyshots,
    temporal_overlap,
)
from src.models import BiLSTMSummarizer
from src.utils import set_seed, setup_logging


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


def _has_any_audio_feature(features_root: Path, video_ids: list[str]) -> bool:
    return any((features_root / f"{video_id}_audio.npy").exists() for video_id in video_ids)


def _infer_bidirectional_from_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
    return any("lstm.weight_hh_l0_reverse" in key for key in state_dict.keys())


def _infer_num_layers_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    indices = []
    for key in state_dict.keys():
        if key.startswith("lstm.weight_ih_l"):
            suffix = key.split("lstm.weight_ih_l", maxsplit=1)[1]
            idx = suffix.replace("_reverse", "")
            if idx.isdigit():
                indices.append(int(idx))
    return max(indices) + 1 if indices else 1


def _infer_hidden_size_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    w_hh = state_dict["lstm.weight_hh_l0"]
    return int(w_hh.shape[1])


def _variant_from_video_id(video_id: str) -> str:
    # TVSum ids in this repo are commonly "video_1"... "video_50".
    if video_id.lower().startswith("video_"):
        return "tvsum"
    return "summe"


def _safe_pct(value: float) -> str:
    return f"{100.0 * value:.1f}"


def _discover_checkpoints(checkpoint_dirs: list[Path]) -> list[Path]:
    checkpoints: list[Path] = []
    for base_dir in checkpoint_dirs:
        if not base_dir.exists():
            print(f"[skip] checkpoint dir not found: {base_dir}")
            continue
        checkpoints.extend(p for p in sorted(base_dir.rglob("*.pt")) if p.is_file())
    return checkpoints


def _build_loader(config, video_ids: list[str]):
    paths = config.paths
    features_root = Path(paths.features_root)
    feature_dim = _feature_dim_from_meta(features_root, config.features.feature_dim)

    if _has_any_audio_feature(features_root, video_ids):
        feature_loader = MultimodalFeatureLoader(
            features_root=features_root,
            visual_dim=feature_dim,
            audio_dim=getattr(config.model, "audio_dim", 384),
            padding_value=config.data.padding_value,
        )
    else:
        feature_loader = FeatureLoader(features_root, feature_dim=feature_dim)
    return feature_loader


def evaluate_checkpoint(
    checkpoint_path: Path,
    config,
    test_loader,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)

    model_input_dim = int(state_dict["lstm.weight_ih_l0"].shape[1])
    hidden_size = _infer_hidden_size_from_state_dict(state_dict)
    bidirectional = _infer_bidirectional_from_state_dict(state_dict)
    num_layers = _infer_num_layers_from_state_dict(state_dict)
    use_attention, fuse_attention = BiLSTMSummarizer.infer_attention_flags_from_state_dict(
        state_dict,
        hidden_size=hidden_size,
        bidirectional=bidirectional,
    )

    model = BiLSTMSummarizer(
        input_dim=model_input_dim,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=config.model.dropout,
        bidirectional=bidirectional,
        use_attention=use_attention,
        fuse_attention_context=fuse_attention,
    )
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    ratio = config.inference.summary_ratio
    min_kf = config.inference.min_keyframes

    buckets: dict[str, dict[str, list[float]]] = {
        "summe": {"p": [], "r": [], "f": [], "o": []},
        "tvsum": {"p": [], "r": [], "f": [], "o": []},
        "all": {"p": [], "r": [], "f": [], "o": []},
    }

    with torch.no_grad():
        for batch in test_loader:
            feats, labels, lengths, video_ids = batch
            feats = feats.to(device)
            lengths = lengths.to(device)

            if feats.size(-1) != model_input_dim:
                if feats.size(-1) < model_input_dim:
                    pad_dim = model_input_dim - feats.size(-1)
                    pad = torch.zeros(feats.size(0), feats.size(1), pad_dim, device=feats.device)
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
                keyframes = select_keyshots(scores[i].cpu().numpy(), length, ratio, min_kf)
                pred_ranges = keyshots_to_ranges(keyframes)
                gt_keyframes = np.where(gt_bin > 0.5)[0].tolist()
                gt_ranges = keyshots_to_ranges(gt_keyframes)
                overlap = temporal_overlap(pred_ranges, gt_ranges, length)

                ds = _variant_from_video_id(str(video_ids[i]))
                for name in (ds, "all"):
                    buckets[name]["p"].append(p)
                    buckets[name]["r"].append(r)
                    buckets[name]["f"].append(f)
                    buckets[name]["o"].append(overlap)

    def _mean(bucket_name: str, key: str) -> float:
        values = buckets[bucket_name][key]
        if not values:
            return 0.0
        return float(np.mean(values))

    return {
        "summe": {
            "fscore": _mean("summe", "f"),
        },
        "tvsum": {
            "fscore": _mean("tvsum", "f"),
        },
        "all": {
            "precision": _mean("all", "p"),
            "recall": _mean("all", "r"),
            "fscore": _mean("all", "f"),
            "temporal_overlap": _mean("all", "o"),
        },
    }


def write_markdown(
    output_path: Path,
    rows: list[tuple[str, Path, dict[str, dict[str, float]]]],
    config_path: Path,
) -> None:
    lines = [
        "# Ablation Results",
        "",
        f"- Generated by `abalation.py`",
        f"- Config: `{config_path.as_posix()}`",
        f"- Total checkpoints: `{len(rows)}`",
        "",
    ]

    for idx, (label, ckpt_path, metrics) in enumerate(rows, start=1):
        lines.extend(
            [
                f"## {idx}. {label}",
                f"- Checkpoint: `{ckpt_path.as_posix()}`",
                "",
                "| SumMe F-score (%) | TVSum F-score (%) | Precision (%) | Recall (%) | Temporal Overlap (%) |",
                "| ---: | ---: | ---: | ---: | ---: |",
                "| "
                + f"{_safe_pct(metrics['summe']['fscore'])} | "
                + f"{_safe_pct(metrics['tvsum']['fscore'])} | "
                + f"{_safe_pct(metrics['all']['precision'])} | "
                + f"{_safe_pct(metrics['all']['recall'])} | "
                + f"{_safe_pct(metrics['all']['temporal_overlap'])} |",
                "",
            ]
        )

    lines.extend(
        [
            "## Notes",
            "- `Precision/Recall/F-score` here follow the current project evaluator.",
            "- GT is binarized by top-k ratio in the existing evaluation logic.",
            "",
        ]
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def write_variant_markdown(
    output_path: Path,
    rows: list[tuple[str, Path, dict[str, dict[str, float]]]],
    config_path: Path,
) -> None:
    lines = [
        "# Ablation Results",
        "",
        f"- Generated by `abalation.py`",
        f"- Config: `{config_path.as_posix()}`",
        "",
        "| Model / Variant | SumMe F-score (%) | TVSum F-score (%) | Precision (%) | Recall (%) | Temporal Overlap (%) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, _, metrics in rows:
        lines.append(
            "| "
            + f"{label} | "
            + f"{_safe_pct(metrics['summe']['fscore'])} | "
            + f"{_safe_pct(metrics['tvsum']['fscore'])} | "
            + f"{_safe_pct(metrics['all']['precision'])} | "
            + f"{_safe_pct(metrics['all']['recall'])} | "
            + f"{_safe_pct(metrics['all']['temporal_overlap'])} |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "- `Precision/Recall/F-score` here follow the current project evaluator.",
            "- GT is binarized by top-k ratio in the existing evaluation logic.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ablation variants and write abalation.md")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--output", type=str, default="abalation.md")
    parser.add_argument(
        "--mode",
        choices=("checkpoints", "variants"),
        default="checkpoints",
        help="checkpoints: one table per checkpoint; variants: one table with fixed ablation rows.",
    )
    parser.add_argument(
        "--checkpoint-dirs",
        type=str,
        nargs="+",
        default=["checkpoints", "checkpoints-optimize"],
        help="Folders to recursively scan .pt checkpoints.",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="*",
        default=[],
        help="Optional explicit checkpoint paths (in addition to --checkpoint-dirs).",
    )
    parser.add_argument("--full", type=str, default="checkpoints/best.pt")
    parser.add_argument("--visual-only", type=str, default="")
    parser.add_argument("--no-attention", type=str, default="")
    parser.add_argument("--unilstm", type=str, default="")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        raise SystemExit(f"Config not found: {config_path.resolve()}")

    config = load_config(str(config_path))
    set_seed(config.seed)
    setup_logging()

    paths = config.paths
    features_root = Path(paths.features_root)
    labels_root = Path(paths.labels_root)
    if not features_root.exists() or not labels_root.exists():
        raise SystemExit("Features or labels directory not found. Check config paths.")

    label_loader = LabelLoader(labels_root)
    video_ids = _collect_video_ids(features_root, labels_root)
    if not video_ids:
        raise SystemExit("No videos with both labels and features were found.")

    feature_loader = _build_loader(config, video_ids)
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
        padding_value=config.data.padding_value,
        seed=config.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows: list[tuple[str, Path, dict[str, dict[str, float]]]] = []

    if args.mode == "variants":
        variants = [
            ("**Multimodal BiLSTM + Temporal Attention** (full model)", Path(args.full)),
            ("Ablation: Visual-only BiLSTM (bỏ audio)", Path(args.visual_only) if args.visual_only else None),
            ("Ablation: BiLSTM không có Temporal Attention", Path(args.no_attention) if args.no_attention else None),
            ("Ablation: UniLSTM (bidirectional = False)", Path(args.unilstm) if args.unilstm else None),
        ]
        for label, ckpt_path in variants:
            if ckpt_path is None:
                continue
            if not ckpt_path.is_file():
                print(f"[skip] checkpoint not found: {ckpt_path}")
                continue
            print(f"[eval] {label} <- {ckpt_path}")
            metrics = evaluate_checkpoint(ckpt_path, config, test_loader, device)
            rows.append((label, ckpt_path, metrics))
    else:
        scan_dirs = [Path(p) for p in args.checkpoint_dirs]
        discovered = _discover_checkpoints(scan_dirs)
        explicit = [Path(p) for p in args.checkpoints if Path(p).is_file()]
        missing_explicit = [Path(p) for p in args.checkpoints if not Path(p).is_file()]
        for p in missing_explicit:
            print(f"[skip] explicit checkpoint not found: {p}")

        all_checkpoints: list[Path] = []
        seen: set[Path] = set()
        for p in discovered + explicit:
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                all_checkpoints.append(p)

        for ckpt_path in all_checkpoints:
            label = f"{ckpt_path.parent.name}/{ckpt_path.name}"
            print(f"[eval] {label} <- {ckpt_path}")
            metrics = evaluate_checkpoint(ckpt_path, config, test_loader, device)
            rows.append((label, ckpt_path, metrics))

    if not rows:
        raise SystemExit("No valid checkpoints found. Check --checkpoint-dirs or --checkpoints.")

    output_path = Path(args.output)
    if args.mode == "variants":
        write_variant_markdown(output_path, rows, config_path)
    else:
        write_markdown(output_path, rows, config_path)
    print(f"[ok] wrote {output_path.resolve()}")


if __name__ == "__main__":
    main()
