## Video Summarization by Supervised Learning  
**Multimodal BiLSTM với Temporal Attention cho tóm tắt video**

---

## 1. Giới thiệu

### 1.1. Bài toán tóm tắt video

- **Tóm tắt video**: tạo ra một phiên bản rút gọn nhưng vẫn giữ lại nội dung và mạch truyện quan trọng nhất.  
- Hai dạng chính:
  - **Tóm tắt tĩnh**: tập các keyframe đại diện.
  - **Tóm tắt động**: đoạn video rút gọn được ghép từ các phân đoạn quan trọng (giữ cả hình và tiếng).  
- Mục tiêu: **giảm thời gian xem** nhưng vẫn nắm được ý chính.

### 1.2. Ví dụ ứng dụng

- **Tình huống hằng ngày**: người dùng có 10 phút nhưng video bài giảng dài 1 giờ. Hệ thống tóm tắt tự động chọn các đoạn giải thích cốt lõi, ví dụ minh họa, kết luận → video rút còn 6–10 phút, vẫn nắm được ~70–80% nội dung.  
- **Góc nhìn giảm dữ liệu**: trong **video classification**, mô hình phải xử lý rất nhiều frame. Nếu chỉ giữ lại ~15% frame quan trọng, ta giảm được chi phí tính toán và lưu trữ cho mô hình phía sau.

### 1.3. Ý nghĩa thực tiễn

- **Tiết kiệm thời gian** cho người xem.  
- **Hỗ trợ gợi ý và truy hồi**: keyframe tốt giúp chọn thumbnail, preview nội dung.  
- **Ứng dụng**: tóm tắt bản tin, bài giảng, MOOC, chương trình TV,…

### 1.4. Ý nghĩa khoa học

- **Giảm chiều có cấu trúc**: thay vì giảm đều, hệ thống chỉ giữ các frame/đoạn có **nhiều thông tin**.  
- **Tiền xử lý cho mô hình downstream**: tóm tắt trước khi huấn luyện phân loại, truy hồi, captioning… giúp giảm chi phí nhưng vẫn giữ phần “giàu thông tin”.

### 1.5. Bối cảnh nghiên cứu

- Các hướng chính:
  - Phương pháp truyền thống: heuristic, clustering, shot detection…
  - Học sâu: LSTM/BiLSTM, attention, Transformer, reinforcement learning, self-supervised.
  - Đa phương thức: kết hợp hình ảnh, âm thanh, transcript, metadata.  
- Một số khoảng trống:
  - Khai thác tốt hơn tín hiệu **text/audio** (lời nói, phụ đề, OCR).
  - Mô hình hóa chuỗi dài bằng kiến trúc phân cấp (hierarchical, transformer).
  - Học unsupervised / semi-supervised để giảm chi phí gán nhãn.

### 1.6. Trọng tâm đề tài

- Bài toán: **multimodal video summarization (hình + tiếng)**, thiết lập **supervised** trên hai bộ dữ liệu benchmark **SumMe** và **TVSum**.  
- Kiến trúc: **BiLSTM với Temporal Attention** (nhánh thị giác) + nhánh audio dùng **Whisper** và **SentenceBERT**.  
- Kết quả chính: **tóm tắt động** (video ngắn kèm âm thanh liền mạch).

---

## 2. Phát biểu bài toán

### 2.1. Input

- Video gốc \(V\) dạng chuẩn (mp4, avi, …), độ dài tùy ý.  
- Trong huấn luyện: với mỗi frame \(t\) có score quan trọng \(y_t \in [0,1]\), được suy ra từ annotation SumMe / TVSum.

### 2.2. Output

- **Tóm tắt tĩnh**: tập keyframe đại diện nội dung chính.  
- **Tóm tắt động**: video rút gọn \(V_{\text{sum}}\) dài khoảng **35%** thời lượng gốc, được ghép từ các đoạn quan trọng, giữ nguyên audio tương ứng.

### 2.3. Tập trung triển khai

- Đề tài ưu tiên **tóm tắt động** vì:
  1. Gần với nhu cầu sử dụng thật (xem video ngắn có tiếng).
  2. Dễ đánh giá định tính bằng cách xem trực tiếp.
  3. Tận dụng đầy đủ pipeline audio dựa trên `ffmpeg`.  
- **Tóm tắt tĩnh (keyframe)** cũng được hỗ trợ như sản phẩm phụ từ bước chọn keyshot.

---

## 3. Phương pháp

### 3.1. Khung học có giám sát

- Chuỗi input: \(\{x_t\}_{t=1}^T, x_t \in \mathbb{R}^{2432}\) là vector **đa phương thức** (hình 2048 + audio 384).  
- Mô hình xuất ra **logits** \(z_t \in \mathbb{R}\) cho mỗi frame; xác suất \(\hat{y}_t = \sigma(z_t)\) được ngầm tính trong loss.  
- Ground truth: score thực \(y_t \in [0,1]\), được **nhị phân hóa** (threshold 0.5) để dùng trong loss.  
- **Loss**: Binary Cross-Entropy with Logits (`BCEWithLogitsLoss`):

\[
\mathcal{L} = -\frac{1}{T}\sum_{t=1}^{T}
  \bigl[y_t \log\sigma(z_t) + (1-y_t)\log(1-\sigma(z_t))\bigr].
\]

### 3.2. Datasets

- **SumMe**: video ngắn, đa dạng chủ đề; annotator đánh dấu đoạn quan trọng → chuyển thành score theo frame.  
- **TVSum**: tập trung nội dung dạng chương trình TV, vlog, documentary; ~20 annotator/ video, lấy trung bình làm ground truth.

### 3.3. Thước đo đánh giá

- **F-score (F-measure)**: overlap giữa tóm tắt sinh ra và tóm tắt tham chiếu.  
- **Precision / Recall** (trên mức frame hoặc segment):  
  - Precision: phần trong tóm tắt thực sự quan trọng.  
  - Recall: phần nội dung quan trọng được giữ lại.  
- **Temporal overlap**: tỷ lệ giao/hiệp trên trục thời gian giữa các segment dự đoán và ground truth.

---

## 4. Kiến trúc đề xuất

### 4.1. Tổng quan

- Hai nhánh song song:
  - Nhánh **video** (trên): Frame Sampling → CNN (ResNet‑50).  
  - Nhánh **audio** (dưới): Audio Extraction → Whisper ASR → BERT Encoding.  
- Ở mỗi frame, hai loại đặc trưng được **concatenate** thành vector \([T \times 2432]\).  
- Chuỗi đặc trưng đưa vào **BiLSTM 2 layer, hidden 256 mỗi chiều** + **Temporal Attention** → logits \(z_t\).  
- Hình kiến trúc trong `proporsal.tex` được vẽ bằng TikZ theo phong cách paper (NeurIPS/ACL).

### 4.2. BiLSTM

- BiLSTM đọc chuỗi theo cả hai chiều thời gian, ẩn trạng thái mỗi frame \(h_t \in \mathbb{R}^{512}\) (256 mỗi hướng).  
- Giúp mô hình hóa ngữ cảnh dài hạn: tầm quan trọng của frame phụ thuộc cả trước và sau nó.

### 4.3. Temporal Attention

- Từ các \(h_t\), mô hình tính attention score \(\alpha_t\) để nhấn mạnh các frame “giàu thông tin”, giảm nhiễu và lặp.  
- Attention kết hợp với BiLSTM giúp tập trung vào vùng thời gian chứa nội dung chính (định nghĩa, kết luận, tiêu đề slide,…).

### 4.4. Output projection

- Một lớp tuyến tính cho mỗi frame: \(z_t = W_o h_t + b_o\).  
- **Không dùng sigmoid trong mạng**, sigmoid được “gộp” vào loss `BCEWithLogitsLoss` cho ổn định số.  
- Khi inference, chuyển logits thành score \(\hat{y}_t = \sigma(z_t)\).

---

## 5. Pipeline xử lý audio

### 5.1. Các bước chính

1. **Audio Extraction**: dùng `ffmpeg` tách audio sang `.wav`.  
2. **ASR (Whisper)**: chuyển audio thành chuỗi câu kèm timestamp \((\text{sentence}_k, t^{start}_k, t^{end}_k)\).  
3. **SentenceBERT**: encode từng câu thành vector \(a^{text}_k \in \mathbb{R}^{384}\).  
4. **Frame-level Alignment**: với mỗi frame \(t\), gán embedding phù hợp dựa trên khoảng thời gian chứa frame đó; nếu không có câu nào, dùng vector 0 hoặc nội suy gần nhất.  
5. **Feature Fusion**: concat với đặc trưng hình ảnh \(v_t \in \mathbb{R}^{2048}\) thành \(x_t = [v_t; a_t] \in \mathbb{R}^{2432}\).

### 5.2. Lợi ích nhánh audio

- Bắt được nội dung **ngữ nghĩa lời nói**: định nghĩa, tiêu đề, kết luận.  
- Bổ sung cho hình ảnh trong các đoạn **ít chuyển động** (slide tĩnh nhưng lời giải thích quan trọng).

---

## 6. Pipeline hệ thống

### 6.1. Pipeline huấn luyện

Bốn phase chính:

1. **Tiền xử lý & trích đặc trưng**  
   - Sample frame 2 fps.  
   - CNN ResNet‑50 → \(v_t \in \mathbb{R}^{2048}\).  
   - Whisper + SBERT → \(a_t \in \mathbb{R}^{384}\).  
   - Lưu `video_id.npy` (visual) và `video_id_audio.npy` (audio).

2. **Xây dựng input & batching**  
   - Tạo chuỗi \(x_t = [v_t; a_t]\).  
   - Cắt đoạn nếu \(T > 960\), pad và mask trong batch.

3. **Forward pass**  
   - Chuỗi \(x_t\) → BiLSTM 2 layer, H=256 → Attention → logits \(z_t\).

4. **Loss & tối ưu**  
   - Loss: `BCEWithLogitsLoss`.  
   - Optimizer: **AdamW**, lr=1e‑3, weight decay=1e‑5.  
   - Scheduler: **StepLR(step\_size=15, gamma=0.5)**.  
   - Early stopping, lưu `checkpoints/best.pt`.

### 6.2. Pipeline suy luận (Inference)

1. Trích đặc trưng \(\{v_t\}, \{a_t\}\) cho video mới (như trong train).  
2. Chạy model → score theo frame.  
3. **Gaussian smoothing** + chuẩn hóa score để giảm nhiễu.  
4. **Shot segmentation** (chia chuỗi thành shot, score theo shot).  
5. **Shot scoring & selection**: chọn các shot tốt nhất sao cho tổng độ dài ≈ \(r \cdot |V|\) với \(r \approx 0.35\).  
6. **Mở rộng segment** thêm ±1.5 giây tạo ngữ cảnh, merge segment gần nhau.  
7. Dùng `ffmpeg` cắt và ghép segment → **video tóm tắt** \(V_{\text{sum}}\) có đầy đủ audio.

---

## 7. Siêu tham số chính

- **Input dim**: 2432 (2048 hình + 384 audio).  
- **BiLSTM**: hidden 256 mỗi chiều, 2 layer, có attention.  
- **Loss**: `BCEWithLogitsLoss`.  
- **Optimizer**: AdamW (lr=0.001, weight decay=1e‑5).  
- **Scheduler**: StepLR (step=15, gamma=0.5).  
- **Sampling**: 2 fps, `max_seq_len`=960 frame (~8 phút).  
- **Summary ratio**: ~35% độ dài gốc.  
- **Mở rộng segment**: ±1.5 giây.

---

## 8. Kết quả kỳ vọng

### 8.1. Mục tiêu định lượng

Trên Split test SumMe và TVSum:

- **F-score**:  
  - SumMe: \(\ge 0.43\)  
  - TVSum: \(\ge 0.55\)  
- **Precision / Recall** tương ứng với mức F-score trên.  
- **Temporal overlap**:  
  - SumMe: \(\ge 0.40\)  
  - TVSum: \(\ge 0.50\).  

Nhánh audio dự kiến tăng **F-score thêm 2–5%** so với BiLSTM chỉ dùng hình.

### 8.2. Kỳ vọng định tính

- Tóm tắt cho video ~10 phút có độ dài **3–4 phút** (~35%).  
- Đoạn được chọn tập trung ở:
  - vùng hoạt động cao, chuyển cảnh quan trọng,
  - thoại giàu nội dung (định nghĩa, kết luận,…).  
- Nhờ smoothing + shot-based selection + mở rộng ±1.5s:
  - video tóm tắt mạch lạc, ít “giật”.  
- Audio giữ nguyên, tránh cắt ngang câu nói.

### 8.3. Kế hoạch ablation

So sánh các biến thể:

- **BiLSTM chỉ hình**: \(v_t \in \mathbb{R}^{2048}\).  
- **Full model (hình + audio)**: \(x_t \in \mathbb{R}^{2432}\).  
- **Không attention**: BiLSTM thuần.  
- **LSTM một chiều**: bỏ backward pass.  

Mục tiêu: đo đóng góp của audio, attention, bidirectionality.

---

## 9. Kế hoạch thực hiện (5 tuần)

- **Tuần 1**: cài môi trường, tải và chuẩn hóa SumMe/TVSum, chốt `config.yaml`.  
- **Tuần 2**: trích đặc trưng hình (ResNet‑50, 2 fps) + audio (Whisper + SBERT), căn chỉnh frame-level.  
- **Tuần 3**: huấn luyện BiLSTM + BCEWithLogitsLoss + AdamW + StepLR; thử nghiệm một số bộ siêu tham số.  
- **Tuần 4**: hoàn thiện pipeline inference (smoothing, shot segmentation, ffmpeg), đánh giá định lượng trên test.  
- **Tuần 5**: chạy ablation, so sánh với baseline trong literature, viết báo cáo và slide.

---

## 10. Hướng phát triển

- **Mở rộng domain**: bản tin, bài giảng, surveillance,… đánh giá riêng cho từng loại video.  
- **Xây dựng bộ dữ liệu “đã giảm chiều”**: dùng video tóm tắt làm input cho bài toán phân loại, captioning, QA trên video.  
- **Tích hợp mô hình downstream**:
  - TimeSformer, Video Swin Transformer, I3D/3D‑CNN,…  
  - So sánh độ chính xác, thời gian train, VRAM khi train trên video gốc vs video đã tóm tắt.  
- **Giả thuyết trung tâm**: tóm tắt video **giảm mạnh kích thước dữ liệu** nhưng vẫn **giữ hiệu năng** cho tác vụ downstream với chi phí tính toán thấp hơn.
