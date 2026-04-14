"""
Streamlit demo: upload video -> summarized video (skim).
Tự động detect input_dim và backbone từ checkpoint — không cần chọn tay.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Video Summarization", page_icon="🎬", layout="centered")

st.title("🎬 Video Summarization Demo")
st.caption("Upload a video or paste a YouTube link → get a short summary video.")


def _ffmpeg_available() -> bool:
    """Check if ffmpeg is installed (for audio in summary video)."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
            timeout=5,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Model")
    config_path     = st.text_input("Config", value="configs/config.yaml")
    checkpoint_path = st.text_input("Checkpoint", value="checkpoints/best.pt")
    summary_ratio   = st.slider("Summary ratio", 0.05, 0.5, 0.15, 0.01)
    skim_fps        = st.slider("Output FPS", 5, 30, 15)
    st.caption("Backbone tự chọn theo checkpoint (1024=GoogLeNet/.h5, 2048=ResNet, 1408=Visual+Audio)")

    if not _ffmpeg_available():
        st.warning(
            "**ffmpeg** chưa cài → video tóm tắt **không có tiếng**. "
            "Cài: [ffmpeg.org](https://ffmpeg.org/download.html) (Windows: `winget install ffmpeg`)"
        )

input_mode = st.radio("Input", ["Upload video", "YouTube URL"], horizontal=True)

if input_mode == "Upload video":
    uploaded = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mkv", "mov"]
    )
    if uploaded is None:
        st.info("Upload a video to start.")
        st.stop()
else:
    youtube_url = st.text_input(
        "YouTube URL", placeholder="https://www.youtube.com/watch?v=..."
    )
    if not youtube_url or ("youtube.com" not in youtube_url and "youtu.be" not in youtube_url):
        st.info("Paste a YouTube link to start.")
        st.stop()

# ── Checkpoint check ──────────────────────────────────────────────────────────
ckpt = Path(checkpoint_path)
if not ckpt.is_absolute():
    ckpt = Path(__file__).resolve().parent / ckpt
if not ckpt.exists():
    st.error(f"Checkpoint not found: {ckpt}. Train a model first.")
    st.stop()
checkpoint_path = str(ckpt)


# ── Auto-detect backbone từ input_dim ─────────────────────────────────────────
def _detect_backbone(input_dim: int) -> str:
    """
    Tự chọn CNN backbone phù hợp với input_dim trong checkpoint.
    
    Mapping:
        1024 → googlenet  (eccv16 .h5 features)
        1408 → googlenet  (1024 visual + 384 audio)
        2048 → resnet50
        2432 → resnet50   (2048 visual + 384 audio)
    """
    visual_dim = input_dim - 384 if input_dim in (1408, 2432) else input_dim
    if visual_dim <= 1024:
        return "googlenet"
    return "resnet50"


def _detect_has_audio(input_dim: int) -> bool:
    """Checkpoint có audio features không (dựa vào input_dim)."""
    return input_dim in (1408, 2432)


# ── Load model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model_and_extractor(_config_path: str, _checkpoint_path: str, _cache_version: int = 2):
    import torch
    from src.config import load_config
    from src.features import CNNFeatureExtractor
    from src.models import BiLSTMSummarizer

    cfg = Path(_config_path)
    if not cfg.is_absolute():
        cfg = Path(__file__).resolve().parent / cfg
    config = load_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Đọc checkpoint
    ckpt_data = torch.load(_checkpoint_path, map_location="cpu")
    state     = ckpt_data["model_state_dict"]

    # Auto-detect input_dim từ weight shape
    # lstm.weight_ih_l0: shape = [4*hidden_size, input_dim]
    input_dim = state["lstm.weight_ih_l0"].shape[1]
    backbone  = _detect_backbone(input_dim)
    has_audio = _detect_has_audio(input_dim)

    model = BiLSTMSummarizer(
        input_dim=input_dim,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        bidirectional=getattr(config.model, "bidirectional", True),
        use_attention=getattr(config.model, "use_attention", True),
    )
    model.load_state_dict(state)
    model = model.to(device).eval()

    extractor = CNNFeatureExtractor(
        backbone=backbone, pretrained=True, device=str(device)
    )
    # Force-sync device for cached objects across reruns.
    extractor.device = device
    extractor._features = extractor._features.to(device)
    extractor._features.eval()

    return model, extractor, device, config, input_dim, backbone, has_audio


# ── YouTube download ──────────────────────────────────────────────────────────
def download_youtube(url: str, out_dir: Path) -> tuple[Path | None, str | None]:
    try:
        import yt_dlp
    except ImportError as e:
        return None, f"yt-dlp not found: {e}. Install: pip install yt-dlp"
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        opts = {
            "format": "best[ext=mp4]/best",
            "outtmpl": str(out_dir / "video.%(ext)s"),
            "quiet": True,
            "no_warnings": True,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([url])
        for f in out_dir.iterdir():
            if f.suffix.lower() in (".mp4", ".mkv", ".webm", ".avi"):
                return f, None
        return None, "Download failed (no video file)."
    except Exception as e:
        err = str(e)
        if any(k in err.lower() for k in ("not available", "private", "unavailable")):
            err = "Video không khả dụng. Thử link khác.\n" + err
        return None, err


# ── Run summary ───────────────────────────────────────────────────────────────
def run_summary(
    video_path: Path,
    model,
    extractor,
    device,
    config,
    has_audio: bool,
    summary_ratio_val: float,
    skim_fps_val: int,
):
    import numpy as np
    import torch
    from src.features import FrameSampler
    from src.evaluation import select_keyshots
    from src.inference.dynamic_summary import create_summary_video, generate_dynamic_summary, export_summary_video

    SAMPLE_RATE = 2
    max_seq_len = getattr(config.data, "max_seq_len", 960)
    sampler = FrameSampler(sample_rate=SAMPLE_RATE)
    frames  = sampler.sample_frames_to_list(video_path)
    if not frames:
        return None, "No frames extracted."

    visual_feats = extractor.extract_from_frames(frames, batch_size=16)
    length       = visual_feats.shape[0]

    if has_audio:
        try:
            from src.features.audio_extractor import AudioExtractor
            audio_ext  = AudioExtractor(
                whisper_model="base",
                device=str(device),
                sample_rate=SAMPLE_RATE,
            )
            audio_feats = audio_ext.extract_from_video(video_path, n_frames=length)
            feats       = np.concatenate([visual_feats, audio_feats], axis=1)
        except Exception as e:
            st.warning(f"⚠️ Audio extraction thất bại ({e}). Dùng visual-only.")
            feats = np.concatenate([visual_feats, np.zeros((length, 384), dtype=np.float32)], axis=1)
    else:
        feats = visual_feats

    with torch.no_grad():
        if length <= max_seq_len:
            x = torch.from_numpy(feats).float().unsqueeze(0).to(device)
            scores, _ = model(x, lengths=torch.tensor([length], device=device))
            scores = scores[0].cpu().numpy()
        else:
            scores_list = []
            for start in range(0, length, max_seq_len):
                end = min(start + max_seq_len, length)
                chunk = feats[start:end]
                x = torch.from_numpy(chunk).float().unsqueeze(0).to(device)
                s, _ = model(x, lengths=torch.tensor([end - start], device=device))
                scores_list.append(s[0].cpu().numpy())
            scores = np.concatenate(scores_list)

    keyframes = select_keyshots(scores, length, summary_ratio=summary_ratio_val, min_keyframes=5)
    out_path = Path(tempfile.gettempdir()) / "streamlit_summary.mp4"

    try:
        create_summary_video(
            video_path=video_path,
            keyframe_indices=keyframes,
            output_path=out_path,
            target_fps=skim_fps_val,
            sample_rate=SAMPLE_RATE,
        )
    except Exception:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        keyframes_original = [
            min(int(round(i * video_fps / SAMPLE_RATE)), total_video_frames - 1)
            for i in keyframes
        ]
        keyframes_original = sorted(set(keyframes_original))
        out_frames = generate_dynamic_summary(video_path, keyframes_original, target_fps=skim_fps_val)
        if not out_frames:
            return None, "No summary frames."
        export_summary_video(out_frames, out_path, fps=skim_fps_val)

    return out_path, None


# ── Generate button ───────────────────────────────────────────────────────────
if st.button("Generate summary", type="primary"):
    video_path = None
    out_path = None
    err = None
    try:
        if input_mode == "Upload video":
            with tempfile.NamedTemporaryFile(
                suffix=Path(uploaded.name).suffix, delete=False
            ) as tmp:
                tmp.write(uploaded.read())
                video_path = Path(tmp.name)
        else:
            yt_dir = Path(tempfile.gettempdir()) / "streamlit_yt_dl"
            with st.spinner("Downloading from YouTube..."):
                video_path, err_dl = download_youtube(youtube_url, yt_dir)
            if err_dl:
                st.error(err_dl)
                st.stop()
            if not video_path or not video_path.exists():
                st.error("Download failed.")
                st.stop()
            single = Path(tempfile.gettempdir()) / f"streamlit_yt_single{video_path.suffix}"
            if single.exists():
                single.unlink()
            import shutil
            shutil.copy2(video_path, single)
            for f in yt_dir.iterdir():
                f.unlink(missing_ok=True)
            video_path = single

        with st.spinner("Loading model..."):
            model, extractor, device, config, input_dim, backbone, has_audio = load_model_and_extractor(
                config_path, checkpoint_path, 2
            )
            st.sidebar.divider()
            st.sidebar.subheader("🔍 Auto-detected")
            st.sidebar.write(f"**Input dim:** {input_dim}")
            st.sidebar.write(f"**Backbone:** {backbone}")
            st.sidebar.write(f"**Audio:** {'✅ có' if has_audio else '❌ không'}")

        with st.spinner("Extracting features and predicting..."):
            out_path, err = run_summary(
                video_path, model, extractor, device, config,
                has_audio, summary_ratio, skim_fps,
            )
    finally:
        if video_path and video_path.exists():
            video_path.unlink(missing_ok=True)

    if err:
        st.error(err)
        if not _ffmpeg_available():
            st.info("💡 Cài **ffmpeg** để export video có tiếng: https://ffmpeg.org/download.html (Windows: `winget install ffmpeg`)")
        st.stop()

    if out_path is not None:
        st.success("Summary video ready.")
        st.video(str(out_path))
        with open(out_path, "rb") as f:
            st.download_button(
                "Download summary video", data=f,
                file_name="summary.mp4", mime="video/mp4",
            )