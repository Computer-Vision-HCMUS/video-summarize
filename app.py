"""
Streamlit demo: upload video -> summarized video (skim).
Uses trained BiLSTM checkpoint; extracts features from uploaded video then runs model.
Imports torch/src only when needed to avoid Streamlit watcher + PyTorch __path__ conflict.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

st.set_page_config(page_title="Video Summarization", page_icon="🎬", layout="centered")

st.title("🎬 Video Summarization Demo")
st.caption("Upload a video or paste a YouTube link → get a short summary video (keyframes only).")

# Sidebar: config and checkpoint
with st.sidebar:
    st.header("Model")
    config_path = st.text_input("Config", value="configs/config.yaml")
    checkpoint_path = st.text_input("Checkpoint", value="checkpoints/best.pt")
    summary_ratio = st.slider("Summary ratio", 0.05, 0.5, 0.15, 0.01)
    skim_fps = st.slider("Output FPS", 5, 30, 15)
    feature_type = st.radio("Feature backbone (must match training)", ["1024 (GoogLeNet / .h5)", "2048 (ResNet)"], index=0)
    input_dim = 1024 if "1024" in feature_type else 2048
    backbone = "googlenet" if input_dim == 1024 else "resnet50"

input_mode = st.radio("Input", ["Upload video", "YouTube URL"], horizontal=True)
video_path_for_run = None

if input_mode == "Upload video":
    uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mkv", "mov"], help="Video file to summarize.")
    if uploaded is None:
        st.info("Upload a video to start.")
        st.stop()
    # Will write to temp file when user clicks Generate
else:
    youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...", help="Uses yt-dlp — install in the same env as Streamlit: pip install yt-dlp")
    if not youtube_url or ("youtube.com" not in youtube_url and "youtu.be" not in youtube_url):
        st.info("Paste a YouTube link to start.")
        st.stop()


def download_youtube(url: str, out_dir: Path) -> tuple[Path | None, str | None]:
    """Download YouTube video to out_dir. Returns (path_to_video, error_message). Error is None on success."""
    try:
        import yt_dlp
    except ImportError as e:
        return None, f"yt-dlp not found in this Python env: {e}. Install: pip install yt-dlp (then restart Streamlit)."
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
        if "not available" in err.lower() or "private" in err.lower() or "unavailable" in err.lower():
            err = "Video không khả dụng (private, đã xóa hoặc giới hạn vùng). Thử link khác hoặc mở link trong trình duyệt để kiểm tra.\n" + err
        return None, err


ckpt = Path(checkpoint_path)
if not ckpt.is_absolute():
    ckpt = Path(__file__).resolve().parent / ckpt
if not ckpt.exists():
    st.error(f"Checkpoint not found: {ckpt}. Train a model first or set the path in the sidebar.")
    st.stop()
checkpoint_path = str(ckpt)

@st.cache_resource
def load_model_and_extractor(_config_path: str, _checkpoint_path: str, _input_dim: int, _backbone: str):
    import torch
    from src.config import load_config
    from src.features import CNNFeatureExtractor
    from src.models import BiLSTMSummarizer
    cfg = Path(_config_path)
    if not cfg.is_absolute():
        cfg = Path(__file__).resolve().parent / cfg
    config = load_config(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMSummarizer(
        input_dim=_input_dim,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        dropout=config.model.dropout,
        bidirectional=getattr(config.model, "bidirectional", True),
        use_attention=getattr(config.model, "use_attention", True),
    )
    ckpt = torch.load(_checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    extractor = CNNFeatureExtractor(backbone=_backbone, pretrained=True, device=str(device))
    return model, extractor, device, config

def run_summary(video_path: Path, model, extractor, device, summary_ratio_val: float, skim_fps_val: int):
    import torch
    from src.features import FrameSampler
    from src.evaluation import select_keyshots
    from src.inference.dynamic_summary import generate_dynamic_summary, export_summary_video
    sampler = FrameSampler(sample_rate=2)
    frames = sampler.sample_frames_to_list(video_path)
    if not frames:
        return None, "No frames extracted."
    feats = extractor.extract_from_frames(frames, batch_size=16)
    length = feats.shape[0]
    x = torch.from_numpy(feats).float().unsqueeze(0).to(device)
    with torch.no_grad():
        scores, _ = model(x, lengths=torch.tensor([length], device=device))
    scores = scores[0].cpu().numpy()
    keyframes = select_keyshots(scores, length, summary_ratio=summary_ratio_val, min_keyframes=5)
    out_frames = generate_dynamic_summary(video_path, keyframes, target_fps=skim_fps_val)
    if not out_frames:
        return None, "No summary frames."
    out_path = Path(tempfile.gettempdir()) / "streamlit_summary.mp4"
    export_summary_video(out_frames, out_path, fps=skim_fps_val)
    return out_path, None

if st.button("Generate summary", type="primary"):
    video_path = None
    try:
        if input_mode == "Upload video":
            with tempfile.NamedTemporaryFile(suffix=Path(uploaded.name).suffix, delete=False) as tmp:
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
            model, extractor, device, config = load_model_and_extractor(config_path, checkpoint_path, input_dim, backbone)
        with st.spinner("Extracting features and predicting..."):
            out_path, err = run_summary(video_path, model, extractor, device, summary_ratio, skim_fps)
    finally:
        if video_path and video_path.exists():
            video_path.unlink(missing_ok=True)

    if err:
        st.error(err)
        st.stop()

    st.success("Summary video ready.")
    st.video(str(out_path))
    with open(out_path, "rb") as f:
        st.download_button("Download summary video", data=f, file_name="summary.mp4", mime="video/mp4")
