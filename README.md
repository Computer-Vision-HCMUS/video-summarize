# Video Summarization — BiLSTM (Supervised)

Production-style Python system for **static** (keyframes) and **dynamic** (video skim) summarization using a BiLSTM with optional temporal attention.

## Project structure

```
video-sumarize/
├── configs/
│   └── config.yaml           # Main config (paths, data, model, training, inference)
├── data/
│   ├── raw/                  # Raw video files
│   ├── features/             # Extracted CNN features (.npy per video)
│   └── labels/               # Frame-level labels (.json per video)
├── src/
│   ├── config/               # Config loader
│   ├── data/                 # Dataset, FeatureLoader, LabelLoader, DataLoader
│   ├── features/             # Frame sampling, CNN extractor, extraction pipeline
│   ├── models/               # BiLSTM, TemporalAttention
│   ├── training/             # Trainer, training/validation loop, checkpointing
│   ├── evaluation/           # F-score, precision/recall, temporal overlap, keyshot selection
│   ├── inference/            # Static/dynamic summary, export video
│   └── utils/                # Logging, seed, type hints
├── scripts/
│   ├── download_datasets.py # Download SumMe+TVSum (Kaggle), prepare labels & videos
│   ├── extract_features.py  # Extract CNN features from videos
│   ├── train.py             # Train model
│   ├── run_eval.py          # Evaluate on test set
│   └── run_inference.py     # Static/dynamic summary from checkpoint
├── app.py                    # Streamlit demo (upload video → summary video)
├── checkpoints/              # Saved model checkpoints
├── logs/                     # Training logs
├── output/                   # Inference outputs
├── notebooks/                # Exploratory notebooks
└── requirements.txt
```

## Quick start (SumMe + TVSum)

**Cách 1 – Chạy tuần tự (tự tải dataset từ Kaggle):**

- **PowerShell (Windows):** `.\run_all.ps1`
- **Bash (Git Bash / WSL / Linux):** `bash run_all.sh` hoặc `bash bash.txt`

Pipeline: cài đặt → tải SumMe+TVSum (Kaggle) → chuẩn bị labels + copy videos vào `data/raw` → extract features → train → eval.

**Cần Kaggle API:** `pip install kaggle`, đặt `kaggle.json` vào `~/.kaggle/` (từ Kaggle Account → Create New API Token).

**Cách 2 – Tải dataset tay rồi chạy:**

1. Tải [SumMe+TVSum từ Kaggle](https://www.kaggle.com/datasets/georgelifinrell/summe-video-summarization), giải nén vào `data/datasets/summe_tvsum/`.
2. Chuẩn bị labels và copy videos:
   ```bash
   python -m scripts.download_datasets --from-dir data/datasets/summe_tvsum --clear-dummy
   ```
3. Extract features và train:
   ```bash
   python -m scripts.extract_features --config configs/config.yaml
   python -m scripts.train --config configs/config.yaml
   python -m scripts.run_eval --config configs/config.yaml --checkpoint checkpoints/best.pt
   ```

**Chỉ dùng .mat (đã có video ở chỗ khác):**  
`python -m scripts.prepare_summe_tvsum --dataset summe --mat /path/to/SumMe.mat --labels-dir data/labels` (rồi tự copy video vào `data/raw` cho đúng `video_id`).

## With real videos (generic)

1. Put videos in `data/raw/`.
2. Add frame-level labels in `data/labels/{video_id}.json`:
   - `{"scores": [0.2, 0.8, ...]}` (float in [0,1]) or
   - `{"keyframes": [0, 5, 10, ...]}` (keyframe indices).
3. Extract features:  
   `python -m scripts.extract_features --config configs/config.yaml [--overwrite]`
4. Train:  
   `python -m scripts.train --config configs/config.yaml`
5. Evaluate:  
   `python -m scripts.run_eval --config configs/config.yaml --checkpoint checkpoints/best.pt`
6. Inference:
   - Static keyframes:  
     `python -m scripts.run_inference --checkpoint checkpoints/best.pt --video-id VID --video-path data/raw/VID.mp4 --static --output-dir output`
   - Dynamic skim:  
     `python -m scripts.run_inference ... --dynamic`

## Streamlit demo

Upload a video or paste a YouTube URL, then download a summarized skim:

```bash
pip install streamlit yt-dlp
python -m streamlit run app.py
```

Use **`python -m streamlit run app.py`** so Streamlit runs in the same Python where you installed yt-dlp (avoids "Install yt-dlp" when it’s already installed in another env). In the sidebar: choose checkpoint path and **feature backbone** (1024 for .h5, 2048 for ResNet). Adjust summary ratio and output FPS if needed.

## Longer videos (nhiều phút hơn)

Config mặc định dùng `max_seq_len: 960` (~8 phút tại 2 fps). Nếu trước đây bạn train với `max_seq_len: 320` (~2.7 phút), cần **train lại** để model học trên đoạn dài hơn:

```bash
python -m scripts.train --config configs/config.yaml
```

Sau khi train xong, demo Streamlit và inference sẽ xử lý video dài theo từng chunk (mỗi chunk 960 frame), rồi gộp điểm để chọn keyframe trên toàn bộ video.

## Config

Edit `configs/config.yaml` for: paths, `max_seq_len`, model `hidden_size`/`num_layers`/`use_attention`, `batch_size`, `learning_rate`, `early_stopping_patience`, `summary_ratio`, etc.

## Reproducibility

`config.seed` is used in `set_seed()` (PyTorch, NumPy, Python random). Training script calls it before building data loaders.
