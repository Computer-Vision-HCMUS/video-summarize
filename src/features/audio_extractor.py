"""
AudioExtractor: trích xuất audio features từ video dùng Whisper + SentenceBERT.

Flow:
    Video → ffmpeg extract audio → Whisper transcribe → segments + timestamps
         → SentenceBERT encode → align với sampled frames → [T, 768] features

Dùng kết hợp với CNNFeatureExtractor:
    visual [T, 1024] + audio [T, 768] → concat [T, 1792] → BiLSTM

Cài đặt:
    pip install openai-whisper sentence-transformers
    pip install ffmpeg-python  (hoặc cài ffmpeg system)
"""

from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

SBERT_DIM   = 384   # all-MiniLM-L6-v2
WHISPER_DIM = 384   # same vì dùng SBERT để encode transcript


class AudioExtractor:
    """
    Trích xuất audio-based features từ video.

    Bước 1: Dùng Whisper để transcribe audio → list of (text, start, end)
    Bước 2: Dùng SentenceBERT encode mỗi segment → vector [384]
    Bước 3: Align theo sampled frame indices → [T, 384] array

    Args:
        whisper_model:  Kích thước Whisper model.
                        "tiny" (nhanh, ít chính xác) | "base" | "small" | "medium"
                        Với video tiếng Việt: dùng "small" hoặc "medium"
        sbert_model:    SentenceTransformer model name.
        device:         "cuda" hoặc "cpu".
        sample_rate:    FPS mà FrameSampler đã dùng (để align frame → timestamp).
                        Default 2 (tức 1 sampled frame = 0.5 giây).
        language:       Ngôn ngữ video. None = Whisper tự detect.
                        "vi" = tiếng Việt, "en" = tiếng Anh.
    """

    def __init__(
        self,
        whisper_model: str = "base",
        sbert_model:   str = "all-MiniLM-L6-v2",
        device:        str = "cpu",
        sample_rate:   int = 2,
        language:      Optional[str] = None,
    ) -> None:
        self.whisper_model_name = whisper_model
        self.sbert_model_name   = sbert_model
        self.device             = device
        self.sample_rate        = sample_rate
        self.language           = language
        self.feature_dim        = SBERT_DIM

        self._whisper = None   # lazy load
        self._sbert   = None   # lazy load

    # ── Lazy loaders ──────────────────────────────────────────────────────────

    def _get_whisper(self):
        if self._whisper is None:
            try:
                import whisper
                logger.info("Loading Whisper model: %s", self.whisper_model_name)
                self._whisper = whisper.load_model(
                    self.whisper_model_name, device="cpu"
                )
            except ImportError:
                raise ImportError(
                    "Whisper chưa cài. Chạy: pip install openai-whisper"
                )
        return self._whisper

    def _get_sbert(self):
        if self._sbert is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info("Loading SentenceTransformer: %s", self.sbert_model_name)
                self._sbert = SentenceTransformer(self.sbert_model_name)
            except ImportError:
                raise ImportError(
                    "sentence-transformers chưa cài. Chạy: pip install sentence-transformers"
                )
        return self._sbert

    # ── Step 1: Extract audio từ video ───────────────────────────────────────

    def _extract_audio(self, video_path: Path, tmp_dir: Path) -> Optional[Path]:
        """Dùng ffmpeg extract audio track → .wav file."""
        audio_path = tmp_dir / "audio.wav"
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),
            "-vn",                    # không lấy video
            "-acodec", "pcm_s16le",   # PCM 16-bit (Whisper yêu cầu)
            "-ar", "16000",           # 16kHz sample rate
            "-ac", "1",               # mono
            "-loglevel", "error",
            str(audio_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode != 0 or not audio_path.exists():
                logger.warning(
                    "ffmpeg extract audio thất bại: %s",
                    result.stderr.decode()
                )
                return None
            return audio_path
        except FileNotFoundError:
            logger.warning("ffmpeg không tìm thấy. Cài: https://ffmpeg.org")
            return None

    # ── Step 2: Transcribe với Whisper ────────────────────────────────────────

    def _transcribe(
        self, audio_path: Path
    ) -> List[Tuple[str, float, float]]:
        """
        Transcribe audio → list of (text, start_sec, end_sec).
        Mỗi segment là 1 câu/cụm từ với timestamp.
        """
        whisper = self._get_whisper()
        kwargs  = {"word_timestamps": False, "verbose": False}
        if self.language:
            kwargs["language"] = self.language

        result   = whisper.transcribe(str(audio_path), **kwargs)
        segments = []
        for seg in result.get("segments", []):
            text  = seg.get("text", "").strip()
            start = float(seg.get("start", 0))
            end   = float(seg.get("end",   0))
            if text:
                segments.append((text, start, end))

        if not segments:
            logger.warning("Whisper không transcribe được nội dung nào.")
        else:
            logger.info("Transcribed %d segments.", len(segments))

        return segments

    # ── Step 3: Encode text segments với SentenceBERT ─────────────────────────

    def _encode_segments(
        self, segments: List[Tuple[str, float, float]]
    ) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
        """
        Encode text → embeddings.
        Returns:
            embeddings:  [N, 384] array
            timestamps:  list of (start, end) cho mỗi embedding
        """
        sbert   = self._get_sbert()
        texts   = [seg[0] for seg in segments]
        times   = [(seg[1], seg[2]) for seg in segments]
        embeds  = sbert.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )   # [N, 384]
        return embeds.astype(np.float32), times

    # ── Step 4: Align features theo frame indices ─────────────────────────────

    def _align_to_frames(
        self,
        embeddings:  np.ndarray,           # [N, 384]
        timestamps:  List[Tuple[float, float]],  # [(start, end), ...]
        n_frames:    int,                   # số sampled frames
    ) -> np.ndarray:
        """
        Align text embeddings → [n_frames, 384] theo timestamp.

        Mỗi sampled frame i tương ứng timestamp = i / sample_rate (giây).
        Frame được assign embedding của segment chứa timestamp đó.
        Nếu frame không nằm trong bất kỳ segment nào → zero vector.

        Args:
            embeddings:  [N, 384] text embeddings.
            timestamps:  [(start_sec, end_sec)] cho mỗi embedding.
            n_frames:    Số sampled frames cần align.

        Returns:
            [n_frames, 384] aligned features.
        """
        aligned = np.zeros((n_frames, self.feature_dim), dtype=np.float32)

        for frame_idx in range(n_frames):
            frame_time = frame_idx / self.sample_rate   # giây trong video gốc

            # Tìm segment chứa frame_time này
            best_embed = None
            best_overlap = 0.0

            for emb, (start, end) in zip(embeddings, timestamps):
                if start <= frame_time <= end:
                    overlap = end - start
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_embed   = emb

            if best_embed is not None:
                aligned[frame_idx] = best_embed
            else:
                # Frame không có speech → interpolate từ segment gần nhất
                min_dist = float("inf")
                nearest  = None
                for emb, (start, end) in zip(embeddings, timestamps):
                    mid  = (start + end) / 2
                    dist = abs(frame_time - mid)
                    if dist < min_dist:
                        min_dist = dist
                        nearest  = emb
                # Chỉ dùng nearest nếu trong vòng 5 giây (tránh assign xa)
                if nearest is not None and min_dist <= 5.0:
                    # Weight giảm dần theo khoảng cách
                    weight = max(0.0, 1.0 - min_dist / 5.0)
                    aligned[frame_idx] = nearest * weight

        return aligned

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_from_video(
        self,
        video_path: str | Path,
        n_frames:   int,
    ) -> np.ndarray:
        """
        Extract audio features từ video → [n_frames, 384] array.

        Args:
            video_path: Path tới video.
            n_frames:   Số sampled frames (phải khớp với CNN features T).

        Returns:
            [n_frames, 384] float32 array.
            Nếu thất bại → zero array (không raise exception, để pipeline tiếp tục).
        """
        video_path = Path(video_path)
        zero       = np.zeros((n_frames, self.feature_dim), dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)

            # 1. Extract audio
            audio_path = self._extract_audio(video_path, tmp)
            if audio_path is None:
                logger.warning("Không extract được audio từ %s. Trả về zero features.", video_path)
                return zero

            # 2. Transcribe
            segments = self._transcribe(audio_path)
            if not segments:
                logger.warning("Không có transcript. Trả về zero features.")
                return zero

            # 3. Encode
            embeddings, timestamps = self._encode_segments(segments)

            # 4. Align
            aligned = self._align_to_frames(embeddings, timestamps, n_frames)

        logger.info(
            "Audio features: %d/%d frames có speech content.",
            int((aligned.sum(axis=1) != 0).sum()), n_frames
        )
        return aligned

    def extract_and_save(
        self,
        video_path: str | Path,
        output_path: str | Path,
        n_frames: int,
        overwrite: bool = False,
    ) -> Optional[Path]:
        """
        Extract + lưu audio features ra file .npy.
        Convention: {video_id}_audio.npy (tách biệt với visual features).
        """
        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            logger.debug("Skip (exists): %s", output_path)
            return output_path
        feats = self.extract_from_video(video_path, n_frames)
        np.save(output_path, feats)
        logger.info("Saved audio features: %s shape=%s", output_path, feats.shape)
        return output_path
