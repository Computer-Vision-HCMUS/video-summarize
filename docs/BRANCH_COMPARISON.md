# So sánh optimize: nhánh enhence-quality vs chienTasks

## Trạng thái Git hiện tại

- **enhence-quality** và **main** trỏ cùng commit `ea587b4` (Merge pull request #3 – merge chienTasks vào enhence-quality).
- **chienTasks** trỏ commit `3d4721d` (trước merge, chứa toàn bộ code được merge).

**Kết luận:** Sau khi merge, **code trên hai nhánh enhence-quality và chienTasks là giống nhau**. Mọi tối ưu/optimize đều nằm trong code chung này (vốn được phát triển trên chienTasks rồi merge vào enhence-quality).

---

## Các optimize có trong codebase (từ chienTasks, hiện có trên cả hai nhánh)

### 1. Model & pipeline

| Nội dung | Mô tả |
|----------|--------|
| **Multimodal (Visual + Audio)** | Thêm audio: Whisper + SentenceBERT → 384-d, concat với visual 1024 → input 1408. Train bằng `train_multimodal.py`. |
| **Auto-detect từ checkpoint** | App đọc `input_dim` từ weight (lstm), suy backbone (googlenet/resnet) và có audio hay không → không cần chọn tay. |
| **Video dài** | Chunk theo `max_seq_len` (960 frame), predict từng chunk rồi gộp score để chọn keyframe toàn video. |

### 2. Frame selection – keyshot (`src/evaluation/keyshot.py`)

| Optimize | Mục đích |
|----------|----------|
| **Gaussian smoothing** | Giảm nhiễu spike đơn lẻ, giữ peak thực sự quan trọng. |
| **Shot-based selection** | Chọn theo đoạn (shot) thay vì frame rời → summary mạch lạc hơn. |
| **Minimum gap** | Tránh nhiều keyframe dồn vào cùng một vị trí. |
| **Diversity enforcement** | Keyframe trải đều theo thời gian, không dồn một đoạn. |

### 3. Export video – dynamic summary (`src/inference/dynamic_summary.py`)

| Optimize | Mục đích |
|----------|----------|
| **ffmpeg thay cv2** | Giữ nguyên audio khi export. |
| **Expand context (±1.5s quanh keyframe)** | Mỗi keyframe thành một đoạn ngắn có ngữ cảnh. |
| **Merge segments gần nhau** | Ghép các đoạn gần nhau → video ít cắt rời. |
| **Fallback cv2** | Khi không có ffmpeg vẫn export được (không audio). |

### 4. App Streamlit (`app.py`)

| Optimize | Mục đích |
|----------|----------|
| **Detect ffmpeg** | Kiểm tra ffmpeg (kể cả Windows PATH), cảnh báo nếu chưa cài. |
| **YouTube temp file duy nhất** | Dùng `tempfile.mkstemp` cho file tải YouTube → tránh FileNotFound khi rerun. |
| **Bắt lỗi run_summary** | Bắt `FileNotFoundError` và exception khác → hiển thị lỗi trên UI thay vì crash. |

---

## So sánh theo “hướng” từ tên nhánh

- **chienTasks:** nhánh làm tính năng và tối ưu (multimodal, keyshot, dynamic export, app).
- **enhence-quality:** nhánh “chất lượng” – sau khi merge chienTasks thì chứa toàn bộ các optimize trên; không có thêm commit riêng khác.

Nếu sau này **enhence-quality** có thêm commit mới (chỉ trên enhence-quality, không có trên chienTasks) thì bảng so sánh optimize sẽ cần cập nhật theo diff giữa hai nhánh tại thời điểm đó.
