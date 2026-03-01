#!/usr/bin/env python3
"""
CLI: Extract CNN features + (optional) Audio features từ raw videos.

Usage:
    # Chỉ visual (như cũ)
    python -m scripts.extract_features --config configs/config.yaml

    # Visual + Audio (multimodal)
    python -m scripts.extract_features --config configs/config.yaml --audio

    # Chỉ audio (nếu visual đã extract rồi)
    python -m scripts.extract_features --config configs/config.yaml --audio-only

Options:
    --audio          Thêm audio extraction sau visual
    --audio-only     Chỉ extract audio (visual đã có)
    --whisper-model  tiny | base | small | medium (default: base)
    --language       Ngôn ngữ video: vi, en, None=auto (default: None)
    --overwrite      Overwrite existing features
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.config import load_config
from src.features import CNNFeatureExtractor, FeatureExtractionPipeline, FrameSampler
from src.features.audio_extractor import AudioExtractor
from src.utils import setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract video features (visual + audio)")
    parser.add_argument("--config",       type=str, default="configs/config.yaml")
    parser.add_argument("--video-dir",    type=str, default=None)
    parser.add_argument("--overwrite",    action="store_true")

    # Audio options
    parser.add_argument("--audio",        action="store_true",
                        help="Extract audio features sau visual")
    parser.add_argument("--audio-only",   action="store_true",
                        help="Chỉ extract audio (bỏ qua visual)")
    parser.add_argument("--whisper-model", type=str, default="base",
                        choices=["tiny", "base", "small", "medium"],
                        help="Whisper model size (lớn hơn = chính xác hơn nhưng chậm hơn)")
    parser.add_argument("--language",     type=str, default=None,
                        help="Ngôn ngữ: 'vi' (tiếng Việt), 'en' (English), None=auto-detect")

    args = parser.parse_args()

    config    = load_config(args.config)
    setup_logging()
    video_dir = Path(args.video_dir or config.paths.data_root)

    if not video_dir.exists():
        raise SystemExit(f"Video directory not found: {video_dir}")

    video_paths = (
        list(video_dir.glob("*.mp4")) +
        list(video_dir.glob("*.avi")) +
        list(video_dir.glob("*.mkv"))
    )
    video_ids = [p.stem for p in video_paths]

    if not video_paths:
        raise SystemExit(f"Không tìm thấy video nào trong {video_dir}")

    print(f"Found {len(video_paths)} videos.")

    # ── Visual extraction ─────────────────────────────────────────────────────
    if not args.audio_only:
        print("\n[1/2] Extracting VISUAL features (CNN)...")
        sampler   = FrameSampler(sample_rate=config.features.sample_rate)
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
        saved = pipeline.process_videos(video_paths, video_ids, overwrite=args.overwrite)
        print(f"✅ Visual features extracted: {len(saved)} videos → {config.paths.features_root}")
    else:
        print("[1/2] Skipping visual extraction (--audio-only mode).")

    # ── Audio extraction ──────────────────────────────────────────────────────
    if args.audio or args.audio_only:
        print(f"\n[2/2] Extracting AUDIO features (Whisper={args.whisper_model}, lang={args.language})...")
        print("      Lần đầu chạy sẽ download Whisper model (~150MB cho 'base').")

        audio_extractor = AudioExtractor(
            whisper_model=args.whisper_model,
            device=config.features.device,
            sample_rate=config.features.sample_rate,
            language=args.language,
        )

        features_root = Path(config.paths.features_root)
        n_success = 0
        n_skip    = 0
        n_fail    = 0

        for vid_path, vid_id in zip(video_paths, video_ids):
            audio_feat_path = features_root / f"{vid_id}_audio.npy"

            if audio_feat_path.exists() and not args.overwrite:
                n_skip += 1
                continue

            # Lấy n_frames từ visual features đã extract
            visual_feat_path = features_root / f"{vid_id}.npy"
            if visual_feat_path.exists():
                import numpy as np
                n_frames = np.load(visual_feat_path).shape[0]
            else:
                # Estimate từ video duration nếu chưa có visual features
                import cv2
                cap      = cv2.VideoCapture(str(vid_path))
                fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
                n_vid_f  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                n_frames = max(1, int(n_vid_f / fps * config.features.sample_rate))

            print(f"  Processing: {vid_id} ({n_frames} frames)...")
            result = audio_extractor.extract_and_save(
                video_path=vid_path,
                output_path=audio_feat_path,
                n_frames=n_frames,
                overwrite=args.overwrite,
            )
            if result is not None:
                n_success += 1
            else:
                n_fail += 1

        print(f"\n✅ Audio features: {n_success} extracted, {n_skip} skipped, {n_fail} failed")
        print(f"   Saved to: {features_root}/ (files: {{video_id}}_audio.npy)")

        if n_success > 0:
            print("\n💡 Để train với audio features, thêm vào config:")
            print("   model:")
            print("     use_audio: true")
            print("     audio_dim: 384")
            print("   Và dùng MultimodalFeatureLoader thay vì FeatureLoader.")
    else:
        print("[2/2] Skipping audio extraction. Thêm --audio để extract audio.")


if __name__ == "__main__":
    main()