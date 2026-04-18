# So sánh nhánh `optimize` với `main`

Tài liệu này tóm tắt các thay đổi trên nhánh **`optimize`** so với **`main`** trong repo `video-summarize` (commit: *Enhance training and model architecture with diversity loss and adaptive learning rate*).

## Tổng quan

| Hạng mục | `main` | `optimize` |
|----------|--------|------------|
| Kiến trúc scorer (attention) | Chỉ dùng output LSTM cho từng frame | Nối **LSTM output** với **context attention toàn video** rồi mới đưa vào scorer |
| Chọn keyframe (inference) | Diversity cứng: chia k segment, mỗi segment 1 shot tốt nhất | **Hybrid 70/30**: một phần top-k theo score, phần còn lại diversity trên shot chưa chọn |
| Hàm mất mát huấn luyện | Chỉ masked BCE | BCE + **diversity/coverage loss** (phạt khi score tập trung một đoạn) |
| Learning rate | `StepLR` (cứ 15 epoch giảm một lần) | **`ReduceLROnPlateau`** theo `val_loss` (patience 5, factor 0.5, min_lr 1e-5) |
| Weight decay | `1e-5` | **`1e-4`** (regularization mạnh hơn) |
| Log / history | `train_loss`, `val_loss` | Thêm `train_bce`, `train_div`, **`lr`** mỗi epoch |

## Các file thay đổi

- `configs/config.yaml` — siêu tham số training mới
- `src/models/bilstm.py` — fuse attention context vào scorer
- `src/evaluation/keyshot.py` — chiến lược hybrid quality + diversity
- `src/training/trainer.py` — scheduler và tham số diversity từ config
- `src/training/loop.py` — loss mới, logging, gọi `scheduler.step(val_loss)` cho Plateau

---

## Fix 1 — Kiến trúc BiLSTM + attention (`bilstm.py`)

**Trên `main`:** Temporal attention tính trọng số nhưng đầu ra quan trọng của frame vẫn chỉ từ `lstm_out[t]`.

**Trên `optimize`:** Context attention `(B, H)` được **broadcast** theo thời gian và **concat** với `out` từng frame → đầu vào scorer có kích thước `2 * out_h` khi bật attention. Mỗi frame được chấm điểm trong bối cảnh **local (LSTM)** và **global (toàn video)**.

---

## Fix 2 — Chọn keyshot hybrid (`keyshot.py`)

**Trên `main`:** `diversity=True` → chia k segment trên toàn bộ shot, mỗi segment chọn 1 shot tốt nhất (diversity “cứng” cho toàn bộ k).

**Trên `optimize`:**

- Tham số `diversity: bool` được thay bằng **`quality_ratio`** (mặc định **0.7**).
- Tính `k_total` như cũ (từ `summary_ratio`, `shot_len`, `min_keyframes`).
- **k_quality** ≈ 70% `k_total`: chọn top shot theo `shot_scores`.
- **k_diversity** ≈ 30% còn lại: chỉ trên các shot **chưa** nằm trong top quality, chia segment và chọn best trong từng segment.
- Gộp, sort, rồi **`_enforce_min_gap`** như trước.

Mục tiêu: tránh chỉ diversity (bỏ lỡ điểm nổi bật) hoặc chỉ top-k (dồn vào một đoạn).

---

## Fix 3 — Diversity / coverage loss (`loop.py` + `config.yaml`)

**Trên `main`:** Chỉ `_masked_bce_loss`.

**Trên `optimize`:** Thêm **`_coverage_diversity_loss`**:

- Với mỗi mẫu trong batch, chia chuỗi hợp lệ thành **`n_segments`** phần (mặc định **5**).
- Mỗi phần lấy **max** của `sigmoid(score)` trên đoạn đó.
- **Penalty = variance** của các max đó → cao khi một vài đoạn có score rất cao và đoạn khác thấp (thiếu coverage).
- Loss diversity = `diversity_weight * mean(variance)` (mặc định **`diversity_weight: 0.3`** trong config).

Tổng loss train/val: **`bce_loss + div_loss`**. History log thêm **`train_bce`**, **`train_div`**.

**Config mới:**

- `training.diversity_weight: 0.3`
- `training.n_segments: 5`
- `weight_decay: 1e-4` (tăng từ `1e-5`)

---

## Fix 4 — Learning rate thích ứng (`trainer.py` + `loop.py`)

**Trên `main`:** `StepLR(optimizer, step_size=15, gamma=0.5)` — giảm LR theo lịch cố định; `scheduler.step()` không đổi.

**Trên `optimize`:** `ReduceLROnPlateau(mode="min", factor=0.5, patience=5, min_lr=1e-5)` — giảm LR khi **`val_loss`** không cải thiện. Trong loop: nếu scheduler là `ReduceLROnPlateau` thì gọi **`scheduler.step(val_loss)`**, ngược lại giữ `scheduler.step()` (tương thích scheduler khác).

Mỗi epoch ghi **`lr`** hiện tại vào `history["lr"]`.

---

## Ghi chú khi merge / tái lập thí nghiệm

- **Inference / eval:** `BiLSTMSummarizer` nhận diện qua shape `scorer.0.weight`: checkpoint cũ (scorer chỉ nhận LSTM `out`, ví dụ 512 chiều) vs checkpoint optimize (concat context, 1024 chiều). `run_eval.py`, `compares-optimize.py`, `app.py` dùng chung logic này.
- **Huấn luyện tiếp / resume:** checkpoint cũ **không** merge trực tiếp vào kiến trúc fuse mới (tham số scorer khác hẳn); cần train từ đầu với code optimize hoặc load có chọn lọc.
- Để chỉnh mức “đều nội dung” vs “đúng điểm GT”, có thể tinh chỉnh `diversity_weight`, `n_segments`, và `quality_ratio` trong `select_keyshots_improved`.
