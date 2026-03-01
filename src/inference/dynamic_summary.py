"""
Dynamic summary: video skim với audio + segment-based export.

Fix 2 vấn đề chính so với version cũ:
  1. Có âm thanh — dùng ffmpeg thay vì cv2.VideoWriter
  2. Xuất theo segments liên tục thay vì frame lẻ rời rạc
     → video mạch lạc hơn, có context, không bị giật cục
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import List, Tuple, Union

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────
# PHẦN 1: Generate segments từ keyframe indices
# ─────────────────────────────────────────────────────────────

def keyframes_to_segments(
    keyframe_indices: List[int],
    total_frames: int,
    fps: float,
    min_seg_seconds: float = 2.0,
    context_seconds: float = 1.0,
    gap_merge_seconds: float = 1.5,
) -> List[Tuple[float, float]]:
    """
    Chuyển danh sách frame indices → list (start_sec, end_sec) segments.

    Khác với version cũ (chỉ lấy đúng frame đó):
    - Mỗi keyframe được expand thêm context_seconds về 2 phía
      → đủ context để hiểu nội dung, không bị cắt giữa chừng
    - Các segments gần nhau (< gap_merge_seconds) được merge lại
      → tránh cắt liên tục vô nghĩa
    - Mỗi segment tối thiểu min_seg_seconds
      → không có đoạn quá ngắn (< 2s) vô nghĩa

    Args:
        keyframe_indices:  Frame indices được model chọn (sampled frame space).
        total_frames:      Tổng số sampled frames.
        fps:               FPS gốc của video (để convert frame → giây).
        min_seg_seconds:   Độ dài tối thiểu mỗi segment (giây).
        context_seconds:   Mở rộng mỗi keyframe ra 2 phía (giây).
        gap_merge_seconds: Khoảng cách tối đa giữa 2 segment để merge.

    Returns:
        List of (start_sec, end_sec) — thời gian trong video gốc.
    """
    if not keyframe_indices or fps <= 0:
        return []

    # Model sample ở sample_rate=2fps → mỗi sampled frame = 0.5s trong video gốc
    # Nhưng ta không biết sample_rate ở đây, nên dùng fps gốc
    # keyframe_indices là index trong sampled frame space
    # Cần biết sample_rate để convert đúng
    # → nhận sample_rate từ caller hoặc estimate từ total_frames vs duration
    # Ở đây giữ đơn giản: coi keyframe_idx như frame index trong video gốc

    half_ctx = context_seconds
    min_gap  = gap_merge_seconds

    # Expand mỗi keyframe thành (start_sec, end_sec)
    raw_segs = []
    for kf in sorted(keyframe_indices):
        t = kf / fps
        start = max(0.0, t - half_ctx)
        end   = t + half_ctx + min_seg_seconds
        raw_segs.append((start, end))

    # Merge overlapping/nearby segments
    merged: List[Tuple[float, float]] = []
    cur_start, cur_end = raw_segs[0]
    for s, e in raw_segs[1:]:
        if s - cur_end <= min_gap:
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))

    # Clamp to video duration
    total_sec = total_frames / fps
    merged = [(s, min(e, total_sec)) for s, e in merged]
    merged = [(s, e) for s, e in merged if e - s >= 0.5]  # loại bỏ quá ngắn

    return merged


# ─────────────────────────────────────────────────────────────
# PHẦN 2: Export với ffmpeg (có âm thanh)
# ─────────────────────────────────────────────────────────────

def _check_ffmpeg() -> bool:
    """Kiểm tra ffmpeg có sẵn không."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def export_segments_with_audio(
    video_path: Union[str, Path],
    segments: List[Tuple[float, float]],
    output_path: Union[str, Path],
    target_fps: int = 15,
) -> Path:
    """
    Cắt video theo segments và ghép lại — CÓ ÂM THANH — dùng ffmpeg.

    Args:
        video_path:   Video gốc.
        segments:     List (start_sec, end_sec).
        output_path:  File output .mp4.
        target_fps:   FPS của video output.

    Returns:
        Path to output video.
    """
    video_path  = Path(video_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not _check_ffmpeg():
        raise RuntimeError(
            "ffmpeg không tìm thấy. Cài đặt: https://ffmpeg.org/download.html\n"
            "Windows: winget install ffmpeg\n"
            "hoặc fallback sang export_summary_video() không có audio."
        )

    if not segments:
        raise ValueError("Không có segments để export.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp = Path(tmp_dir)
        clip_paths = []

        # Cắt từng segment
        for i, (start, end) in enumerate(segments):
            duration = end - start
            if duration < 0.1:
                continue
            clip_path = tmp / f"clip_{i:04d}.mp4"
            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", str(video_path),
                "-t", str(duration),
                "-vf", f"fps={target_fps}",
                "-c:v", "libx264",
                "-c:a", "aac",        # giữ audio
                "-avoid_negative_ts", "make_zero",
                "-loglevel", "error",
                str(clip_path),
            ]
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode == 0 and clip_path.exists():
                clip_paths.append(clip_path)

        if not clip_paths:
            raise RuntimeError("Không cắt được clip nào từ video.")

        if len(clip_paths) == 1:
            # Chỉ có 1 clip → copy thẳng
            import shutil
            shutil.copy2(clip_paths[0], output_path)
            return output_path

        # Tạo file list cho ffmpeg concat
        list_file = tmp / "clips.txt"
        with open(list_file, "w") as f:
            for cp in clip_paths:
                f.write(f"file '{cp}'\n")

        # Ghép tất cả clips lại
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c:v", "libx264",
            "-c:a", "aac",
            "-loglevel", "error",
            str(output_path),
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg concat thất bại:\n{result.stderr.decode()}"
            )

    return output_path


# ─────────────────────────────────────────────────────────────
# PHẦN 3: Fallback không có ffmpeg (giữ backward compatible)
# ─────────────────────────────────────────────────────────────

def generate_dynamic_summary(
    video_path: Union[str, Path],
    keyframe_indices: List[int],
    target_fps: int = 15,
) -> List[np.ndarray]:
    """
    Backward compatible với code cũ.
    Vẫn trả về list frames nhưng lấy theo segments thay vì frame lẻ.
    """
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    segments = keyframes_to_segments(
        keyframe_indices, total_frames, fps,
        min_seg_seconds=2.0,
        context_seconds=1.0,
        gap_merge_seconds=1.5,
    )

    # Convert segments → set of frame indices để extract
    selected_frames_set = set()
    for start_sec, end_sec in segments:
        start_f = int(start_sec * fps)
        end_f   = int(end_sec * fps)
        for f in range(start_f, end_f):
            selected_frames_set.add(f)

    frames = []
    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx in selected_frames_set:
                frames.append(frame)
            frame_idx += 1
    finally:
        cap.release()

    return frames


def export_summary_video(
    frames: List[np.ndarray],
    output_path: Union[str, Path],
    fps: int = 15,
    codec: str = "mp4v",
) -> Path:
    """
    Backward compatible — export frames không có audio.
    Dùng khi ffmpeg không có sẵn.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("No frames to export")
    h, w    = frames[0].shape[:2]
    fourcc  = cv2.VideoWriter_fourcc(*codec)
    writer  = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()
    return output_path


# ─────────────────────────────────────────────────────────────
# PHẦN 4: Hàm chính — tự động chọn ffmpeg hay fallback
# ─────────────────────────────────────────────────────────────

def create_summary_video(
    video_path: Union[str, Path],
    keyframe_indices: List[int],
    output_path: Union[str, Path],
    target_fps: int = 15,
    sample_rate: int = 2,
) -> Path:
    """
    Hàm chính để tạo summary video — CÓ AUDIO nếu ffmpeg available.

    Dùng hàm này thay cho generate_dynamic_summary + export_summary_video cũ.

    Args:
        video_path:        Video gốc.
        keyframe_indices:  Sampled frame indices từ model.
        output_path:       Output .mp4 path.
        target_fps:        FPS output.
        sample_rate:       FPS mà FrameSampler đã sample (default 2fps).
                           Dùng để convert sampled frame idx → video timestamp.

    Returns:
        Path to output video.
    """
    video_path  = Path(video_path)
    output_path = Path(output_path)

    # Lấy thông tin video
    cap         = cv2.VideoCapture(str(video_path))
    video_fps   = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_vid_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # Sampled frame idx → timestamp trong video gốc
    # Mỗi sampled frame tương ứng 1/sample_rate giây
    timestamps = [idx / sample_rate for idx in keyframe_indices]
    total_sec  = total_vid_f / video_fps

    # Convert timestamps → frame indices trong video gốc để tính segments
    video_frame_indices = [int(t * video_fps) for t in timestamps]

    segments = keyframes_to_segments(
        video_frame_indices,
        total_frames=total_vid_f,
        fps=video_fps,
        min_seg_seconds=2.0,
        context_seconds=1.5,
        gap_merge_seconds=2.0,
    )

    if not segments:
        raise ValueError("Không tạo được segments từ keyframes.")

    # Dùng ffmpeg nếu có (có audio), fallback sang cv2 nếu không
    if _check_ffmpeg():
        return export_segments_with_audio(
            video_path, segments, output_path, target_fps
        )
    else:
        print("⚠️  ffmpeg không tìm thấy → export không có audio.")
        print("   Cài ffmpeg để có audio: https://ffmpeg.org/download.html")
        frames = generate_dynamic_summary(video_path, video_frame_indices, target_fps)
        return export_summary_video(frames, output_path, target_fps)