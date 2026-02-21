#!/usr/bin/env python3
"""
Download SumMe and TVSum, export labels to data/labels, copy videos to data/raw.

Cách 1 - Tự động (cần Kaggle API):
  pip install kaggle
  Đặt kaggle.json vào ~/.kaggle/ (từ Kaggle Account -> Create New API Token)
  python -m scripts.download_datasets --all

Cách 2 - Tải tay từ Kaggle rồi giải nén:
  - SumMe+TVSum: https://www.kaggle.com/datasets/georgelifinrell/summe-video-summarization
  - Giải nén vào data/datasets/summe_tvsum/
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
from src.utils import setup_logging

setup_logging()
LOG = __import__("logging").getLogger(__name__)

# Kaggle dataset slugs
KAGGLE_SUMME_TVSUM = "georgelifinrell/summe-video-summarization"
DATASETS_DIR = Path("data/datasets")
LABELS_DIR = Path("data/labels")
RAW_VIDEOS_DIR = Path("data/raw")
VIDEO_EXTENSIONS = (".mp4", ".avi", ".mkv", ".mov")


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
    """Từ thư mục đã giải nén: tìm .mat, export labels, copy videos sang data/raw."""
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
    """Xóa toàn bộ dummy features và labels."""
    for d, pattern in [(Path("data/features"), "dummy_*"), (Path("data/labels"), "dummy_*")]:
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
        prepare_from_directory(Path(args.from_dir))
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
