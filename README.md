# Video Summarization — BiLSTM Multimodal (Supervised)

Hệ thống tóm tắt video tự động dùng **BiLSTM + Temporal Attention** với multimodal features (visual + audio), được train theo hướng supervised learning trên dataset SumMe và TVSum.

## Cách hoạt động

### Tổng quan pipeline

```
RAW VIDEO
   │
   ├─► [1. Extract Features]
   │         ├─► CNN (GoogLeNet) → visual features [T, 1024]
   │         └─► Whisper + SentenceBERT → audio features [T, 384]
   │                    concat → [T, 1408]
   │
   ├─► [2. Train]
   │         ├─► BiLSTM học từ features + ground truth labels
   │         └─► Labels = đánh giá của 15-20 người xem video (TVSum/SumMe)
   │
   └─► [3. Summarize]
             ├─► Extract features từ video mới
             ├─► BiLSTM predict importance score mỗi frame
             ├─► Chọn keyframes (shot-based + diversity)
             └─► ffmpeg ghép thành summary video có audio
```

### Visual Features (CNN)

FrameSampler lấy 2 frame/giây từ video → GoogLeNet extract đặc trưng hình ảnh (màu sắc, texture, hình dạng) → vector `[1024]` mỗi frame. Kết quả: `{video_id}.npy` shape `[T, 1024]`.

### Audio Features (Whisper + SentenceBERT)

ffmpeg extract audio → Whisper transcribe thành các câu kèm timestamp → SentenceBERT encode mỗi câu thành vector `[384]` → align theo frame timestamp. Frame nào có lời thoại → embedding của câu đó, frame im lặng → zero vector. Kết quả: `{video_id}_audio.npy` shape `[T, 384]`.

### Supervised Learning

Model học cách **bắt chước đánh giá của con người**: nhìn vào features của frame → predict con người sẽ cho frame đó bao nhiêu điểm quan trọng. TVSum/SumMe labels được tạo bởi 15-20 annotators xem video và đánh dấu đoạn nào quan trọng → average thành score `[0, 1]` mỗi frame. BiLSTM học statistical pattern giữa (visual + audio features) và label score.

---

## Project Structure

```
video-summarize/
├── configs/
│   └── config.yaml                   # Config chính (paths, model, training, inference)
├── data/
│   ├── raw/                          # Raw video files (.mp4, .avi, ...)
│   ├── features/                     # Extracted features
│   │   ├── {video_id}.npy            # Visual features [T, 1024]
│   │   ├── {video_id}_audio.npy      # Audio features [T, 384]
│   │   └── _meta.json                # Feature dim metadata
│   ├── labels/                       # Ground truth labels
│   │   └── {video_id}.json           # Frame-level importance scores [T]
│   └── datasets/                     # Downloaded datasets (gitignore)
├── src/
│   ├── config/                       # Config loader
│   ├── data/
│   │   ├── dataset.py                # VideoSummarizationDataset
│   │   ├── dataloader.py             # DataLoader factory
│   │   ├── feature_loader.py         # Visual FeatureLoader
│   │   ├── multimodal_loader.py      # Visual + Audio loader (MỚI)
│   │   ├── label_loader.py           # Label loader
│   │   └── summe_tvsum.py            # SumMe/TVSum parser
│   ├── features/
│   │   ├── frame_sampler.py          # Sample frames từ video (2fps)
│   │   ├── cnn_extractor.py          # GoogLeNet/ResNet feature extractor
│   │   ├── audio_extractor.py        # Whisper + SentenceBERT (MỚI)
│   │   └── pipeline.py               # End-to-end extraction pipeline
│   ├── models/
│   │   ├── bilstm.py                 # BiLSTMSummarizer
│   │   └── attention.py              # TemporalAttention
│   ├── training/
│   │   ├── trainer.py                # Trainer (build model + optimizer)
│   │   └── loop.py                   # Training/validation loop
│   ├── evaluation/
│   │   ├── keyshot.py                # Shot-based keyframe selection (CẢI TIẾN)
│   │   └── metrics.py                # F-score, precision, recall, temporal overlap
│   ├── inference/
│   │   ├── dynamic_summary.py        # Segment-based export có audio (CẢI TIẾN)
│   │   └── static_summary.py         # Keyframe export
│   └── utils/                        # Logging, seed
├── scripts/
│   ├── download_datasets.py          # Download SumMe+TVSum
│   ├── extract_features.py           # Extract visual + audio features
│   ├── train.py                      # Train visual-only
│   ├── train_multimodal.py           # Train visual + audio (MỚI)
│   ├── run_eval.py                   # Evaluate F-score
│   └── run_inference.py              # Static/dynamic summary
├── app.py                            # Streamlit demo (auto-detect model config)
├── checkpoints/                      # Model checkpoints
├── logs/                             # Training logs
└── output/                           # Inference outputs
```

---

## Cài đặt

```bash
pip install -r requirements.txt

# Thêm cho multimodal (audio)
pip install openai-whisper sentence-transformers

# ffmpeg (để export video có audio)
# Windows:
winget install ffmpeg
# Mac:
brew install ffmpeg
```

---

## Quick Start

### Cách 1 — Dùng pre-extracted features (.h5) — Nhanh nhất (Không có audio)

Không cần raw videos, dùng luôn features từ eccv16:

```bash
# Download features từ Kaggle
python -m scripts.download_datasets --all

# Train visual-only
python -m scripts.train --config configs/config.yaml

# Evaluate
python -m scripts.run_eval --config configs/config.yaml --checkpoint checkpoints/best.pt
```

### Cách 2 — Dùng raw videos + multimodal (Visual + Audio)

```bash
# Bước 1: Download raw videos (SumMe ~2.2GB + TVSum ~670MB)
# Tải từ Zenodo: https://zenodo.org/record/4884870
# Giải nén vào data/raw/

# Bước 2: Extract visual + audio features
python -m scripts.extract_features --config configs/config.yaml --audio --whisper-model base

# Với video tiếng Việt
python -m scripts.extract_features --audio --whisper-model small --language vi

# Bước 3: Train multimodal
python -m scripts.train_multimodal --config configs/config.yaml

# Bước 4: Evaluate
python -m scripts.run_eval --config configs/config.yaml --checkpoint checkpoints/best.pt
```

## Streamlit Demo

```bash
python -m streamlit run app.py
```

App tự động detect từ checkpoint:
- `input_dim` → không cần chọn backbone tay
- Có audio hay không → tự extract audio khi inference
- Hiển thị thông tin model ở sidebar sau khi load

Sidebar settings:
- **Checkpoint**: đường dẫn tới file `.pt`
- **Summary ratio**: tỷ lệ video được giữ lại (0.15 = 15%)
- **Output FPS**: FPS của video output

---

## So sánh Model

| | Visual Only | Visual + Audio |
|---|---|---|
| Input dim | 1024 | 1408 |
| Hiểu hình ảnh | ✅ | ✅ |
| Hiểu lời nói | ❌ | ✅ |
| Output có audio | ✅ (ffmpeg) | ✅ (ffmpeg) |
| Checkpoint | `best.pt` (input_dim=1024) | `best.pt` (input_dim=1408) |
| Train script | `scripts/train.py` | `scripts/train_multimodal.py` |

---

## Cải tiến so với baseline

**Frame selection (`src/evaluation/keyshot.py`):**
- Gaussian smoothing trên scores → loại nhiễu spike
- Shot-based selection → chọn đoạn có nghĩa thay vì frame lẻ
- Diversity enforcement → keyframes trải đều video
- Minimum gap → tránh chọn nhiều frames cùng vị trí

**Export (`src/inference/dynamic_summary.py`):**
- Dùng ffmpeg thay cv2 → giữ nguyên audio track
- Expand context ±1.5 giây quanh keyframe → đủ ngữ cảnh
- Merge segments gần nhau → video mạch lạc hơn
- Fallback sang cv2 nếu ffmpeg chưa cài

**App (`app.py`):**
- Tự detect `input_dim` và backbone từ checkpoint
- Tự extract audio khi inference nếu model được train với audio

---

## Config

Chỉnh `configs/config.yaml` cho các thông số:

```yaml
model:
  hidden_size: 256      # BiLSTM hidden size
  num_layers: 2         # Số LSTM layers
  dropout: 0.3
  use_attention: true   # Temporal attention

training:
  epochs: 50
  batch_size: 8
  learning_rate: 0.001
  early_stopping_patience: 10

inference:
  summary_ratio: 0.15   # 15% video được giữ lại
  min_keyframes: 5
```

---

## Reproducibility

`config.seed` được dùng trong `set_seed()` (PyTorch, NumPy, Python random) — đảm bảo kết quả reproducible giữa các lần train.