#!/usr/bin/env python3
"""
Download SumMe and TVSum — mọi thứ cài trong thư mục project (data/..., đã gitignore).

- Zip + bản giải nén: data/datasets/
- Labels JSON: data/labels/
- Video copy để train: data/raw/

Cách 1 - Tự động (cần Kaggle API):
  pip install kaggle
  Đặt kaggle.json vào ~/.kaggle/ (từ Kaggle Account -> Create New API Token)
  python -m scripts.download_datasets --all

Cách 2 - Tải tay từ Kaggle rồi giải nén vào thư mục project:
  - SumMe+TVSum: https://www.kaggle.com/datasets/georgelifinrell/summe-video-summarization
  - Giải nén vào <project>/data/datasets/summe_tvsum/
  python -m scripts.download_datasets --from-dir data/datasets/summe_tvsum
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

from src.data.summe_tvsum import (
    load_summe_mat,
    load_tvsum_mat,
    export_labels_to_json,
)
from src.data.h5_summe_tvsum import prepare_from_h5
from src.utils import setup_logging

setup_logging()
LOG = __import__("logging").getLogger(__name__)

# Tất cả dataset nằm trong thư mục project (gitignore)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = PROJECT_ROOT / "data" / "datasets"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
RAW_VIDEOS_DIR = PROJECT_ROOT / "data" / "raw"
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov")

# Kaggle dataset slugs
KAGGLE_SUMME_TVSUM = "georgelifinrell/summe-video-summarization"


def _run(cmd: list[str], cwd: Path | None = None) -> bool:
    try:
        subprocess.run(cmd, check=True, cwd=cwd, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        LOG.warning("Command failed: %s", e)
        return False


def download_from_kaggle() -> Path | None:
    """Download combined SumMe+TVSum from Kaggle. Returns extracted dir or None."""
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = DATASETS_DIR / "summe-video-summarization.zip"
    if not zip_path.exists():
        ok = _run(["kaggle", "datasets", "download", "-p", str(DATASETS_DIR), KAGGLE_SUMME_TVSUM])
        if not ok:
            LOG.error(
                "Kaggle download failed. Install: pip install kaggle. "
                "Place kaggle.json in ~/.kaggle/ (from Kaggle Account -> Create New API Token)."
            )
            return None
    extract_dir = DATASETS_DIR / "summe_tvsum"
    if zip_path.exists():
        extract_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(extract_dir)
        LOG.info("Extracted to %s", extract_dir)
    return extract_dir if extract_dir.exists() else None


def find_mat_and_videos(root: Path):
    """Yield (path_to_mat, 'summe'|'tvsum', path_to_videos_dir)."""
    root = Path(root)
    for mat_path in root.rglob("*.mat"):
        name = mat_path.name.lower()
        if "summe" in name:
            kind = "summe"
        elif "tvsum" in name or "tv_sum" in name:
            kind = "tvsum"
        else:
            continue
        # Videos thường cùng thư mục với .mat hoặc trong Videos/ videos/
        video_dir = mat_path.parent
        for sub in ("Videos", "videos", "Video", "video"):
            d = mat_path.parent / sub
            if d.is_dir():
                video_dir = d
                break
        yield mat_path, kind, video_dir


def collect_video_by_stem(video_dir: Path) -> dict[str, Path]:
    """Map filename stem -> full path (first match)."""
    out = {}
    for ext in VIDEO_EXTENSIONS:
        for f in video_dir.glob(f"*{ext}"):
            stem = f.stem
            if stem not in out:
                out[stem] = f
    # Có thể video nằm trong subdir
    for sub in video_dir.iterdir():
        if sub.is_dir():
            for ext in VIDEO_EXTENSIONS:
                for f in sub.glob(f"*{ext}"):
                    stem = f.stem
                    if stem not in out:
                        out[stem] = f
    return out


def prepare_from_directory(data_root: Path) -> None:
    """From extracted dir: prefer .h5 (eccv16) -> labels + features; else .mat -> labels + copy videos."""
    data_root = Path(data_root)
    # 1) Try eccv16 .h5 first (SumMe/TVSum with pre-extracted features)
    n_h5 = prepare_from_h5(data_root, LABELS_DIR, FEATURES_DIR, meta_path=FEATURES_DIR / "_meta.json")
    if n_h5 > 0:
        LOG.info("Prepared %d videos from .h5 (labels + features). Skip extract_features.", n_h5)
        return

    # 2) Fallback: .mat + copy videos to data/raw (then run extract_features)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
    all_pairs = []
    video_paths_map = {}

    for mat_path, kind, video_dir in find_mat_and_videos(data_root):
        if kind == "summe":
            pairs = load_summe_mat(mat_path)
        else:
            pairs = load_tvsum_mat(mat_path)
        if not pairs:
            continue
        all_pairs.extend(pairs)
        by_stem = collect_video_by_stem(video_dir)
        for vid, _ in pairs:
            if vid in by_stem:
                video_paths_map[vid] = by_stem[vid]
            # Một số .mat dùng tên có khoảng trắng / khác extension
            for stem, path in by_stem.items():
                if stem.replace(" ", "_") == vid or stem == vid.replace(" ", "_"):
                    video_paths_map[vid] = path
                    break

    if not all_pairs:
        LOG.warning("No (video_id, scores) loaded. Check .mat structure under %s", data_root)
        return

    export_labels_to_json(all_pairs, LABELS_DIR)
    LOG.info("Exported %d labels to %s", len(all_pairs), LABELS_DIR)

    copied = 0
    for vid, _ in all_pairs:
        src = video_paths_map.get(vid)
        if not src or not src.exists():
            continue
        dest = RAW_VIDEOS_DIR / f"{vid}{src.suffix}"
        if dest.resolve() != src.resolve():
            shutil.copy2(src, dest)
            copied += 1
    LOG.info("Copied %d videos to %s", copied, RAW_VIDEOS_DIR)


def clear_dummy_data() -> None:
    """Xóa toàn bộ dummy features và labels (trong thư mục project)."""
    for d, pattern in [
        (PROJECT_ROOT / "data" / "features", "dummy_*"),
        (PROJECT_ROOT / "data" / "labels", "dummy_*"),
    ]:
        if not d.exists():
            continue
        for f in d.glob(pattern):
            f.unlink()
            LOG.info("Removed %s", f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download SumMe & TVSum, prepare labels and videos")
    parser.add_argument("--all", action="store_true", help="Download from Kaggle then prepare")
    parser.add_argument("--from-dir", type=str, default=None, help="Use existing extracted dir (no download)")
    parser.add_argument("--clear-dummy", action="store_true", help="Remove dummy_* from data/features and data/labels")
    args = parser.parse_args()

    if args.clear_dummy:
        clear_dummy_data()
        if not args.from_dir and not args.all:
            return

    if args.from_dir:
        from_dir = Path(args.from_dir)
        if not from_dir.is_absolute():
            from_dir = PROJECT_ROOT / from_dir
        prepare_from_directory(from_dir)
        return

    if args.all:
        clear_dummy_data()
        extracted = download_from_kaggle()
        if extracted:
            prepare_from_directory(extracted)
        return

    parser.print_help()
    print("\nVí dụ: --all (tải Kaggle) hoặc --from-dir data/datasets/summe_tvsum (đã giải nén)")


if __name__ == "__main__":
    main()
