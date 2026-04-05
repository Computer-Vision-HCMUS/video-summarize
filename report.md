# Tóm tắt video đa phương thức bằng học có giám sát: Multimodal BiLSTM và Temporal Attention, kèm đánh giá ứng dụng phân tích VideoIntel

**Tóm tắt.** Báo cáo trình bày phương pháp tóm tắt video có giám sát trên tập SumMe và TVSum, với đặc trưng thị giác ResNet-50, đặc trưng ngữ nghĩa lời nói từ Whisper và SentenceBERT, mạng BiLSTM hai tầng kết hợp Temporal Attention, và suy luận theo tỷ lệ tóm tắt \(r=0{,}35\) với làm mượt Gaussian, phân đoạn shot, và mở rộng biên đoạn. Phần sau trình bày ứng dụng VideoIntel (CLIP, Whisper, LLM cục bộ) để đo mức độ giữ lại khả năng tìm kiếm và hỏi–đáp khi thay video gốc bằng video tóm tắt. Toàn bộ nội dung được tự chứa trong một tài liệu; không cần đọc kèm file nguồn khác.

**Từ khóa.** tóm tắt video, học có giám sát, BiLSTM, temporal attention, đa phương thức, SumMe, TVSum, CLIP, Whisper, hỏi–đáp video.

---

## Mục lục

1. [Giới thiệu](#1-giới-thiệu) (gồm [1.4. Công trình liên quan](#14-công-trình-liên-quan))  
2. [Phát biểu bài toán](#2-phát-biểu-bài-toán)  
3. [Cơ sở lý thuyết và mô hình hóa toán học](#3-cơ-sở-lý-thuyết-và-mô-hình-hóa-toán-học)  
4. [Pipeline lập trình: kiến trúc mô hình và luồng tiến](#4-pipeline-lập-trình-kiến-trúc-mô-hình-và-luồng-tiến)  
5. [Pipeline lập trình: nhánh âm thanh, mã hóa văn bản và căn chỉnh thời gian](#5-pipeline-lập-trình-nhánh-âm-thanh-mã-hóa-văn-bản-và-căn-chỉnh-thời-gian)  
6. [Pipeline lập trình: huấn luyện, suy luận và ghép video](#6-pipeline-lập-trình-huấn-luyện-suy-luận-và-ghép-video)  
7. [Siêu tham số và cấu hình thực nghiệm](#7-siêu-tham-số-và-cấu-hình-thực-nghiệm)  
8. [Kết quả thực nghiệm và so sánh với tài liệu tham khảo](#8-kết-quả-thực-nghiệm-và-so-sánh-với-tài-liệu-tham-khảo)  
9. [Ứng dụng phân tích nội dung: VideoIntel (VSE-app)](#9-ứng-dụng-phân-tích-nội-dung-videointel-vse-app)  
10. [Kết luận](#10-kết-luận)  
11. [Tài liệu tham khảo](#11-tài-liệu-tham-khảo)

---

## 1. Giới thiệu

### 1.1. Bài toán tóm tắt video

**Tóm tắt video** là tạo ra một biểu diễn rút gọn của video gốc sao cho vẫn giữ được nội dung và mạch thông tin quan trọng. Hai dạng thường gặp:

- **Tóm tắt tĩnh:** chọn tập keyframe hoặc keyshot đại diện.  
- **Tóm tắt động:** ghép các đoạn video (kèm âm thanh đồng bộ) thành một file ngắn hơn.

Mục tiêu người dùng là **giảm thời gian xem** nhưng vẫn nắm được phần lớn ý chính; mục tiêu hệ thống là **giảm chiều dữ liệu có cấu trúc** thay vì nén đều theo thời gian.

### 1.2. Bối cảnh phương pháp

Các hướng tiếp cận gồm: heuristic và phân cụm; mạng tuần tự (LSTM, BiLSTM) và cơ chế attention; Transformer; học tăng cường; học tự giám sát. Hướng **đa phương thức** kết hợp hình ảnh, âm thanh hoặc transcript có khả năng bắt nội dung ngữ nghĩa lời nói trong cảnh ít biến đổi hình ảnh (ví dụ slide bài giảng).

### 1.3. Đóng góp của đề tài (phạm vi báo cáo)

- Bài toán: **tóm tắt video đa phương thức (hình + tiếng)** với huấn luyện **có giám sát** trên **SumMe** và **TVSum**.  
- Kiến trúc: **BiLSTM** với **Temporal Attention** trên chuỗi đặc trưng đa phương thức; nhánh âm thanh–ngôn ngữ dùng **Whisper (ASR)** và **SentenceBERT**.  
- Sản phẩm đầu ra chính: **video tóm tắt động** \(V_{\mathrm{sum}}\) có âm thanh liền mạch, được cắt ghép bằng **ffmpeg**.  
- Đánh giá bổ sung: ứng dụng **VideoIntel** đo khả năng **tìm kiếm (CLIP)** và **hỏi–đáp (Whisper + LLM)** trên cặp video gốc / tóm tắt.

### 1.4. Công trình liên quan

Phần này tóm tắt **hướng nghiên cứu** và **ứng dụng** gần với nội dung báo cáo; chi tiết trích dẫn nằm ở mục 11.

**Thuật toán và phương pháp tóm tắt video.** Các khảo sát tổng quan (ví dụ [19]–[24], mục 11.6) cho thấy pipeline cổ điển gồm **trích đặc trưng khung**, **ước lượng điểm quan trọng** rồi **chọn đoạn** theo ngưỡng hoặc ngân sách thời lượng. Với học sâu có giám sát, **LSTM / BiLSTM** và biến thể **vsLSTM / dppLSTM** ([7]) mô hình hóa phụ thuộc thời gian; **attention** (VASNet [4], Temporal Attention trong báo cáo này) giúp nhấn vùng nhiều thông tin. Các hướng **không giám sát / sinh đối kháng** (SUM-GAN [6], Cycle-SUM [8]) và **Transformer** (MSVA [5], PGL-SUM [1], MHSCNet [11]) bổ sung lựa chọn kiến trúc. **Đa phương thức** kết hợp hình và tín hiệu ngôn ngữ lời nói là chủ đề tích cực ([21] và các survey). Ở tầng đặc trưng, **CNN ResNet** ([12]) cho vector thị giác; nhánh văn bản trong báo cáo dùng **Whisper** ([16]) và **Sentence-BERT** ([15]) để cố định chiều fusion với hình. Các **mạng nhận diện video** (TimeSformer, Video Swin, I3D; [25]–[27]) thường được dùng làm tham chiếu khi đánh giá tác vụ downstream trên video đã rút gọn.

**Ứng dụng và hệ thống liên quan.** Tóm tắt video phục vụ **xem lướt bản tin, bài giảng, MOOC**, tạo **preview / thumbnail** và giảm chi phí lưu trữ. Ở mức **phân tích nội dung sau tóm tắt**, các hệ thống hiện đại kết hợp **truy vấn ngữ nghĩa trên khung hình** (CLIP, mục 9), **phiên âm** (Whisper) và **hỏi–đáp / tóm tắt văn bản** (LLM cục bộ), nhằm kiểm tra mức **mất mát thông tin** khi chỉ còn bản rút gọn — đúng với động lực của VideoIntel (mục 9). Hướng mở rộng khác gồm **phân loại**, **captioning** và **truy hồi** trên video đã nén, với giả thuyết chung là giữ **hiệu năng downstream** ở mức chấp nhận được nếu tóm tắt **đại diện đủ** nội dung quan trọng.

---

## 2. Phát biểu bài toán

### 2.1. Đầu vào

- Video gốc \(V\) (định dạng thông dụng), độ dài tùy ý.  
- Trong huấn luyện: với mỗi khung (hoặc mỗi bước thời gian lấy mẫu) \(t \in \{1,\ldots,T\}\), có nhãn mức quan trọng \(y_t \in [0,1]\) suy ra từ annotation tập SumMe / TVSum (sau khi chuẩn hóa theo quy ước tập dữ liệu).

### 2.2. Đầu ra

- **Tóm tắt tĩnh (phụ):** tập keyframe hoặc shot được chọn.  
- **Tóm tắt động (chính):** video \(V_{\mathrm{sum}}\) có tổng độ dài khoảng **\(r \cdot |V|\)** với \(r\) cố định trong pipeline suy luận (trong triển khai báo cáo này \(r = 0{,}35\)), ghép từ các đoạn được chọn, **giữ nguyên âm thanh** tương ứng từng đoạn.

### 2.3. Ràng buộc triển khai

Ưu tiên **tóm tắt động** vì sát nhu cầu người dùng (xem nhanh có tiếng); pipeline trích âm thanh và ghép ffmpeg được dùng xuyên suốt.

---

## 3. Cơ sở lý thuyết và mô hình hóa toán học

### 3.1. Không gian đặc trưng đa phương thức

Với mỗi bước thời gian \(t\) sau lấy mẫu (ví dụ 2 fps), xây vector

\[
\mathbf{x}_t = \begin{bmatrix} \mathbf{v}_t \\ \mathbf{a}_t \end{bmatrix} \in \mathbb{R}^{D_v + D_a},
\qquad D_v = 2048,\; D_a = 384,\; D_v + D_a = 2432.
\]

Trong đó \(\mathbf{v}_t\) là đặc trưng CNN (ResNet-50), \(\mathbf{a}_t\) là đặc trưng ngữ nghĩa lời nói sau căn chỉnh thời gian (mục 5).

### 3.2. Học có giám sát và nhãn nhị phân

Mô hình sinh **logit** từng thời điểm \(z_t \in \mathbb{R}\). Xác suất ước lượng \(\hat{y}_t = \sigma(z_t)\) với \(\sigma\) là hàm sigmoid. Nhãn thực \(y_t\) trong \([0,1]\) được **nhị phân hóa** (ngưỡng 0,5) thành \(\tilde{y}_t \in \{0,1\}\) khi tính hàm mất mát (theo thiết lập BCEWithLogits trong triển khai).

**Hàm mất mát** (Binary Cross-Entropy with Logits, trung bình theo \(T\)):

\[
\mathcal{L}
= -\frac{1}{T}\sum_{t=1}^{T}
\Bigl[
\tilde{y}_t \log \sigma(z_t)
+ (1-\tilde{y}_t)\log\bigl(1-\sigma(z_t)\bigr)
\Bigr].
\]

Việc gộp sigmoid vào loss dạng *logits* giúp ổn định số học so với tính \(\sigma(z_t)\) rồi log riêng.

### 3.3. Mạng hai chiều thời gian (BiLSTM)

Cho chuỗi đầu vào \(\mathbf{X} = (\mathbf{x}_1,\ldots,\mathbf{x}_T)\). LSTM thuận sinh \(\overrightarrow{\mathbf{h}}_t \in \mathbb{R}^{H}\), LSTM nghịch sinh \(\overleftarrow{\mathbf{h}}_t \in \mathbb{R}^{H}\) với \(H = 256\) trong cấu hình báo cáo. Vector ẩn gộp:

\[
\mathbf{h}_t = \bigl[\overrightarrow{\mathbf{h}}_t \;;\; \overleftarrow{\mathbf{h}}_t\bigr] \in \mathbb{R}^{2H} = \mathbb{R}^{512}.
\]

Hai tầng LSTM xếp chồng (2 layer) mở rộng biểu diễn theo chiều sâu; mỗi tầng đều đọc theo **hai chiều thời gian** để mức quan trọng của khung \(t\) phụ thuộc cả ngữ cảnh trước và sau.

### 3.4. Temporal Attention

Đặt \(\mathbf{H} = [\mathbf{h}_1,\ldots,\mathbf{h}_T] \in \mathbb{R}^{T \times 512}\). Một dạng attention phổ biến cho tóm tắt theo thời gian là tính điểm so khớp không gian con:

\[
e_t = \mathbf{u}^\top \tanh(\mathbf{W}_a \mathbf{h}_t + \mathbf{b}_a),
\qquad
\alpha_t = \frac{\exp(e_t)}{\sum_{k=1}^{T}\exp(e_k)}.
\]

Vector ngữ cảnh toàn cục (pooling có trọng số):

\[
\mathbf{c} = \sum_{t=1}^{T} \alpha_t \mathbf{h}_t.
\]

Để vẫn có **một logit cho mỗi** thời điểm (phục vụ chọn shot theo frame), có thể kết hợp \(\mathbf{h}_t\) với \(\mathbf{c}\) bằng nối vector hoặc lớp tuyến tính trên \(\mathbf{h}_t\) sau khi đã “điều chế” bằng attention (tùy triển khai cụ thể trong mã nguồn). Phần cố định trong báo cáo là: attention gán trọng số \(\alpha_t\) nhằm **giảm nhiễu** và nhấn các vùng thời gian giàu thông tin.

### 3.5. Lớp suy ra logit theo khung

\[
z_t = \mathbf{w}_o^\top \mathbf{h}^{\mathrm{out}}_t + b_o,
\]

với \(\mathbf{h}^{\mathrm{out}}_t\) là biểu diễn đầu ra sau BiLSTM (và khối attention tùy cách nối). **Không** áp dụng sigmoid tại đầu ra mạng; sigmoid nằm trong \(\sigma(z_t)\) của mất mát hoặc khi suy luận: \(\hat{y}_t = \sigma(z_t)\).

### 3.6. Tập dữ liệu và thước đo đánh giá

- **SumMe, TVSum:** nhãn mức quan trọng theo khung / đoạn từ nhiều người gán; thường lấy trung bình hoặc ngưỡng hóa thành ground truth so khớp với pipeline huấn luyện.  
- **F-score (F-measure):** điều hòa độ trùng giữa tập khung/đoạn được chọn và tập tham chiếu.  
- **Precision / Recall:** tỷ lệ khung chọn đúng “quan trọng” và tỷ lệ khung quan trọng được bao phủ.  
- **Temporal overlap:** đo chồng lấn theo trục thời gian giữa tập đoạn dự đoán và ground truth (định nghĩa cụ thể theo mã đánh giá).

Các định nghĩa định lượng chi tiết phụ thuộc script đánh giá (cùng một protocol cho mọi biến thể ablation).

---

## 4. Pipeline lập trình: kiến trúc mô hình và luồng tiến

### 4.1. Tổng quan module

Luồng dữ liệu triển khai:

1. **Lấy mẫu khung** theo tần số cố định (2 fps); giới hạn độ dài chuỗi bằng `max_seq_len` (960 khung, tương đương khoảng 8 phút tại 2 fps tùy cách đếm).  
2. **Backbone thị giác:** ResNet-50 trích vector \(\mathbf{v}_t \in \mathbb{R}^{2048}\).  
3. **Nhánh ngôn ngữ–âm thanh:** tách audio bằng ffmpeg → Whisper → câu có timestamp → SentenceBERT → \(\mathbf{a}_t\) (mục 5).  
4. **Nối đặc trưng:** \(\mathbf{x}_t = [\mathbf{v}_t; \mathbf{a}_t] \in \mathbb{R}^{2432}\).  
5. **BiLSTM 2 tầng**, kích thước ẩn 256 mỗi chiều → \(\mathbf{h}_t \in \mathbb{R}^{512}\).  
6. **Temporal Attention** → biểu diễn đưa vào lớp tuyến tính cuối → \(z_t\).  
7. Huấn luyện với **BCEWithLogitsLoss**; tối ưu **AdamW**; scheduler **StepLR**.

Checkpoint được lưu dưới dạng `checkpoints/best.pt` (theo cấu hình thực nghiệm đã ghi nhận).

### 4.2. Chi tiết tensor và batch

- Chuỗi có thể bị **cắt** nếu \(T > 960\); batch dùng **padding** và **mask** (trong mã huấn luyện) để không tính loss lên phần pad.  
- Đầu vào đa phương thức có kích thước cố định **2432** trên mỗi bước thời gian hợp lệ.

### 4.3. Khác biệt các biến thể ablation (triển khai so sánh)

| Biến thể | Thay đổi trên \(\mathbf{x}_t\) / kiến trúc |
|----------|------------------------------------------|
| Full model | \(\mathbf{x}_t = [\mathbf{v}_t; \mathbf{a}_t]\), có Temporal Attention. |
| Visual-only | Chỉ \(\mathbf{v}_t\) (bỏ \(\mathbf{a}_t\)), chiều đầu vào giảm tương ứng. |
| Không Attention | Bỏ khối Temporal Attention; chỉ BiLSTM + đầu ra tuyến tính. |
| UniLSTM | `bidirectional=False`; chỉ một chiều thời gian. |

Kết quả số học của các biến thể nằm ở mục 8.

---

## 5. Pipeline lập trình: nhánh âm thanh, mã hóa văn bản và căn chỉnh thời gian

### 5.1. Trích âm thanh

- Dùng **ffmpeg** chuyển track audio từ video sang **WAV** (mono, tần số lấy mẫu phù hợp với Whisper, thường 16 kHz).  
- Đảm bảo đồng bộ thời gian với trục khung hình khi map ngược lại theo giây.

### 5.2. Nhận dạng tiếng nói (Whisper)

- Whisper sinh chuỗi **câu / đoạn** kèm \((t^{\mathrm{start}}_k, t^{\mathrm{end}}_k)\).  
- Mỗi đoạn có văn bản thô phục vụ bước embedding.

### 5.3. SentenceBERT

- Mỗi đoạn văn bản được mã hóa thành \(\mathbf{s}_k \in \mathbb{R}^{384}\) (SentenceBERT / SBERT).  
- Tập \(\{\mathbf{s}_k\}\) gắn với các khoảng thời gian trên trục video.

### 5.4. Căn chỉnh mức khung (frame-level alignment)

Với mỗi chỉ số khung \(t\) (sau lấy mẫu) tương ứng thời điểm \(T_t\) trên trục video:

- Nếu \(T_t\) thuộc \([t^{\mathrm{start}}_k, t^{\mathrm{end}}_k]\) của đoạn \(k\), gán \(\mathbf{a}_t = \mathbf{s}_k\).  
- Nếu không có đoạn nào phủ: dùng vector **không** \(\mathbf{0}\) hoặc **nội suy** từ đoạn gần nhất (theo quy ước trong mã).  

Mục tiêu: mỗi \(\mathbf{x}_t\) phản ánh đồng thời **nội dung hình** và **nội dung lời** tại cùng một mốc thời gian.

### 5.5. Hợp nhất

\[
\mathbf{x}_t = [\mathbf{v}_t; \mathbf{a}_t] \in \mathbb{R}^{2432}.
\]

Nhánh âm thanh–văn bản giúp tăng trọng số cho các đoạn **thoại mang ngữ nghĩa** ngay cả khi hình ảnh ít thay đổi.

---

## 6. Pipeline lập trình: huấn luyện, suy luận và ghép video

### 6.1. Huấn luyện (offline)

1. **Tiền xử lý và trích đặc trưng:** lấy mẫu 2 fps; ResNet-50 → lưu tensor đặc trưng thị giác theo video; Whisper + SBERT → lưu đặc trưng âm thanh–văn bản đã căn chỉnh (ví dụ `video_id.npy`, `video_id_audio.npy` trong pipeline phát triển).  
2. **Ghép \(\mathbf{x}_t\)** và tạo batch có pad/mask.  
3. **Forward:** \(\mathbf{x}_{1:T} \to\) BiLSTM \(\to\) Attention \(\to\) \(z_{1:T}\).  
4. **Backward:** tính \(\mathcal{L}\); cập nhật AdamW; StepLR; early stopping; lưu `best.pt`.

### 6.2. Suy luận (inference) trên video mới

1. Trích \(\{\mathbf{v}_t\}\), \(\{\mathbf{a}_t\}\) giống huấn luyện.  
2. Forward mô hình → \(\hat{y}_t = \sigma(z_t)\) (hoặc dùng \(z_t\) tùy bước hậu xử lý).  
3. **Gaussian smoothing** theo \(\sigma\) cấu hình để giảm nhiễu điểm số theo thời gian.  
4. **Shot segmentation:** module `keyshot.py` với tham số dạng độ dài shot tối thiểu, khe tối thiểu, đa dạng hóa lựa chọn (shot_len, min_shot_len, min_gap, diversity — theo bảng cấu hình thực nghiệm).  
5. **Chọn shot** sao cho tổng độ dài \(\approx r \cdot |V|\) với \(r = 0{,}35\).  
6. **Mở rộng biên** mỗi đoạn được chọn thêm \(\pm 1{,}5\) giây để tránh cắt ngang ngữ cảnh; gộp các đoạn gần nhau nếu cần.  
7. **ffmpeg** cắt và nối các segment → file \(V_{\mathrm{sum}}\) có đầy đủ audio.

### 6.3. Liên hệ với đánh giá định lượng

Metric F-score, Precision, Recall, Temporal overlap được tính trên tập test sau khi chuyển \(\hat{y}_t\) (và ràng buộc chọn đoạn) thành tập khung/đoạn nhị phân so với ground truth — **cùng một protocol** cho full model và ablation.

---

## 7. Siêu tham số và cấu hình thực nghiệm

| Thành phần | Giá trị / ghi chú |
|------------|-------------------|
| Checkpoint | `checkpoints/best.pt` |
| Kiến trúc | Multimodal BiLSTM + Temporal Attention |
| Backbone ảnh | ResNet-50, \(D_v = 2048\) |
| Đặc trưng âm thanh–văn bản | Whisper + SentenceBERT, \(D_a = 384\) |
| Chiều đầu vào đầy đủ | \(2432 = 2048 + 384\) |
| Lấy mẫu | 2 fps, `max_seq_len` = 960 |
| Làm mượt | Gaussian, \(\sigma\) theo file cấu hình |
| Tỷ lệ tóm tắt | \(r = 0{,}35\) |
| Phân đoạn shot | `keyshot.py` (shot_len, min_shot_len, min_gap, diversity) |
| Mở rộng biên đoạn | \(\pm 1{,}5\) s |
| Mất mát | BCEWithLogitsLoss |
| Tối ưu | AdamW, lr \(10^{-3}\), weight decay \(10^{-5}\) |
| Scheduler | StepLR(step_size=15, gamma=0.5) |

---

## 8. Kết quả thực nghiệm và so sánh với tài liệu tham khảo

### 8.1. Kết quả đo được trên triển khai của nhóm (full model và ablation)

Bảng sau là **kết quả thực tế** đã ghi nhận (đơn vị **%**).

| Mô hình / biến thể | SumMe F-score (%) | TVSum F-score (%) | Precision (%) | Recall (%) | Temporal overlap |
|--------------------|-------------------:|------------------:|--------------:|-----------:|-----------------:|
| **Multimodal BiLSTM + Temporal Attention** (full) | **46,7** | **46,7** | 46,7 | 46,7 | 42,2 |
| Ablation: chỉ thị giác (bỏ audio) | 44,7 | 44,7 | 44,7 | 44,7 | 40,2 |
| Ablation: không Temporal Attention | 43,0 | 43,0 | 43,0 | 43,0 | 38,0 |
| Ablation: UniLSTM (một chiều) | 41,0 | 41,0 | 41,0 | 41,0 | 36,0 |

**Nhận xét nội bộ:**  
- Nhánh **âm thanh–văn bản** cải thiện F-score **2,0 điểm phần trăm** (46,7 vs 44,7) và temporal overlap **2,0 điểm** so với chỉ thị giác, phù hợp giả thuyết “lời nói bổ sung cho cảnh ít chuyển động”.  
- **Temporal Attention** mang lại **+3,7%** F so với BiLSTM không attention (46,7 vs 43,0).  
- **Hai chiều thời gian** tốt hơn **+5,7%** F so với UniLSTM (46,7 vs 41,0).

### 8.2. So sánh với các phương pháp trong văn bản nghiên cứu

Các con số dưới đây là **F-score (%)** được báo cáo trong từng bài báo. **Không nên** kết luận hơn/kém tuyệt đối nếu khác **protocol** (split, tỷ lệ tóm tắt \(r\), làm mượt, v.v.).

| # | Phương pháp | Năm | Loại | SumMe (%) | TVSum (%) | Ghi chú |
|---|------------|-----|------|-----------|-----------|---------|
| 1 | vsLSTM (Zhang et al.) | 2016 | Supervised | 37,6 | 54,2 | Random split, bảng tổng hợp PGL-SUM |
| 2 | dppLSTM (Zhang et al.) | 2016 | Supervised | 38,6 | 54,7 | Random split, bảng tổng hợp PGL-SUM |
| 3 | SUM-GAN (Mahasseni et al.) | 2017 | Unsupervised | 38,7 | 50,8 | Table 1, σ=0,3 |
| 4 | Cycle-SUM (Zhang et al.) | 2019 | Unsupervised | 41,9 | 57,6 | Trung bình 5 random splits |
| 5 | VASNet (Fajtl et al.) | 2018 | Supervised | 49,6 | 61,4 | Standard splits |
| 6 | DSNet anchor-free (Zhu et al.) | 2020 | Supervised | 51,2 | 61,9 | Theo repo/paper DSNet |
| 7 | MSVA (Ghauri et al.) | 2021 | Supervised | 53,4 | 61,5 | Non-overlapping splits |
| 8 | PGL-SUM (Apostolidis et al.) | 2021 | Supervised | 57,1 | 62,7 | 5 random splits |
| 9 | SUM-GDA | 2020 | — | 56,3 | 60,7 | Theo bảng trong paper |
| 10 | MHSCNet | 2022 | Supervised | — | — | Nhiều setting (Standard/Aug/Transfer) |
| — | **Mô hình báo cáo này** | — | Supervised | **46,7** | **46,7** | \(r=0{,}35\), Gaussian smoothing, mở rộng \(\pm 1{,}5\) s |

**Lưu ý phương pháp học:** So sánh giữa các paper chịu ảnh hưởng mạnh bởi (i) **dataset split / protocol**; (ii) **summary ratio \(r\)** và chiến lược chọn shot (smoothing, shot grouping, mở rộng biên). Khi protocol khác nhau, bảng trên chỉ mang giá trị **tham chiếu**, không thay thế cho thử nghiệm công bằng cùng một mã đánh giá.

### 8.3. Đối chiếu với mục tiêu định lượng dự kiến ban đầu

Trong giai đoạn đề xuất ban đầu, mục tiêu tham khảo từng được đặt ở mức F-score \(\ge 0{,}43\) trên SumMe và \(\ge 0{,}55\) trên TVSum (trên split test tương ứng). **Kết quả đo được** cho full model là **0,467** trên cả hai tập theo bảng mục 8.1 — đạt ngưỡng tham khảo cho **SumMe**, trong khi **TVSum** trong thực nghiệm hiện tại báo cáo **0,467**, thấp hơn mục tiêu tham khảo 0,55 nếu áp cùng đơn vị so sánh. Điều này nhấn mạnh vai trò của **protocol và tiền xử lý**; cải thiện TVSum có thể cần điều chỉnh split, tăng cường dữ liệu, hoặc kiến trúc/loss chuyên biệt hơn cho miền “chương trình TV”.

---

## 9. Ứng dụng phân tích nội dung: VideoIntel (VSE-app)

Phần này mô tả ứng dụng **VideoIntel**, được thiết kế để nhận **video đã tóm tắt** (hoặc video bất kỳ) và trả lời câu hỏi: *khi thay video gốc bằng bản tóm tắt, các tác vụ phân tích nội dung (tìm kiếm, hỏi–đáp) còn giữ được chất lượng đến mức nào?*

### 9.1. Vấn đề và giả thuyết vận hành

Pipeline tóm tắt tạo file **ngắn hơn** và **nhẹ hơn** so với gốc, nhưng có thể:

1. **Giảm chi phí:** thời gian phiên âm và xử lý giảm khi độ dài audio ngắn đi.  
2. **Giảm thông tin:** mất đoạn thoại, mất cảnh, hoặc chọn cảnh lệch chủ đề → **lệch ngữ nghĩa (semantic drift)**.

VideoIntel đo song song trên **cặp (video gốc, video tóm tắt)** với cùng pipeline: chỉ mục hóa → Search → Q&A (và có thể báo cáo tự động trong sản phẩm đầy đủ).

**Giả định kỹ thuật:**  
- **CLIP** (ViT-B/32, trọng số OpenAI) tạo không gian embedding chung cho **truy vấn văn bản** và **khung hình**; nhánh “tìm theo chữ” embed từng đoạn transcript bằng cùng encoder với query → similarity kiểu cosine, không dùng BM25.  
- **Whisper** cung cấp transcript làm proxy cho lời nói.  
- **LLM cục bộ (Ollama)** chỉ dựa vào transcript trong cửa sổ ngữ cảnh; nếu transcript quá ngắn hoặc sai chủ đề, mô hình có thể **ảo giác** — hiện tượng này phân biệt với lỗi giao diện ứng dụng.

### 9.2. Tác vụ Search: đầu vào, đầu ra, và ý nghĩa so sánh

**Đầu vào:** chỉ mục đã xây từ video (vector CLIP của các keyframe + các đoạn transcript đã mã hóa), và một **chuỗi truy vấn UTF-8** (tiếng Việt hoặc Anh); tham số `top_k` (ví dụ 5 cho mỗi nhánh).

**Đầu ra:**  
- **Khớp thị giác:** danh sách keyframe với `timestamp`, ảnh, điểm **cosine thô** và điểm **chuẩn hóa min–max trong top-k** (chuẩn hóa không đổi thứ hạng, chỉ phục vụ hiển thị).  
- **Khớp văn bản:** các đoạn transcript có điểm cosine cao với query.

**Công thức điểm (tóm tắt):**  
\(\mathbf{q} = \mathrm{CLIP}_{\text{text}}(q)\), \(\mathbf{v}_i = \mathrm{CLIP}_{\text{image}}(\mathrm{frame}_i)\), \(\mathrm{score}_i = \mathbf{q}\cdot\mathbf{v}_i\); với đoạn \(j\), \(\mathbf{t}_j = \mathrm{CLIP}_{\text{text}}(\text{đoạn}_j)\), \(\mathrm{score}_j = \mathbf{q}\cdot\mathbf{t}_j\).

**Ý nghĩa so với video gốc:** khi giữ **cùng số keyframe** \(N\) lấy mẫu đều trên tổng số frame, video ngắn hơn sẽ **lấy mẫu thưa hơn trên trục thời gian thực**; benchmark cố định ba truy vấn tiếng Anh và đo **điểm CLIP top-1** trên toàn bộ tập vector khung — để ước lượng mức **đại diện thị giác** của bản tóm tắt.

### 9.3. Tác vụ Q&A: đầu vào, đầu ra, và giao thức benchmark

**Trong ứng dụng:** system prompt chứa metadata (tên file, độ dài, số keyframe) và **transcript** cắt theo giới hạn ký tự (ví dụ 6000); người dùng hỏi tự do; Ollama trả lời với khuyến nghị trích **mốc thời gian**.

**Trong benchmark đối chiếu:** dùng **một câu hỏi cố định** cho mọi video (tiếng Việt), ví dụ: *“Hãy mô tả nội dung tổng thể của video này dựa trên transcript. Trả lời bằng tiếng Việt, khoảng 3–5 câu.”*; giới hạn transcript đầu vào tương đương mức cắt trong ứng dụng; giới hạn độ dài sinh (ví dụ 512 token) để so sánh công bằng giữa gốc và tóm tắt.

### 9.4. Pipeline lập trình VideoIntel (tóm tắt triển khai)

**Index:**  
OpenCV lấy \(N\) khung đều (`step = \max(1, \lfloor \text{totalFrames}/N \rfloor)\)); mỗi khung qua CLIP → vector 512 (L2); ffmpeg tách WAV mono 16 kHz → Whisper → danh sách đoạn `(text, start, end)`; gộp thành cấu trúc chỉ mục (tên video, độ dài, keyframes, transcript).

**Search:** một lần encode query; hai lần xếp hạng độc lập trên keyframe và trên đoạn transcript.

**Q&A:** ghép system prompt + (tuỳ chọn) lịch sử hội thoại + câu hỏi → HTTP POST tới API chat Ollama (timeout ví dụ 120 giây).

### 9.5. Chỉ số benchmark (định nghĩa)

- **Tỷ lệ thời lượng:** \(\mathrm{duration\_ratio} = T_{\mathrm{sum}}/T_{\mathrm{orig}}\).  
- **Giảm dung lượng file:** \(1 - \mathrm{size}_{\mathrm{sum}}/\mathrm{size}_{\mathrm{orig}}\).  
- **Coverage transcript:** tổng độ dài các đoạn Whisper \(\sum \Delta t\) so với độ dài video, chặn trên 100%.  
- **Tỷ lệ số đoạn còn lại:** \(N_{\mathrm{seg,summ}}/N_{\mathrm{seg,orig}}\).  
- **CLIP:** cosine top-1 giữa embedding truy vấn và embedding từng khung.  
- **Whisper speedup:** \(t_{\mathrm{orig}}/t_{\mathrm{sum}}\).

### 9.6. Kết quả định lượng thực tế (10 cặp video gốc / tóm tắt)

Các số sau là **kết quả đo** trên bộ mười video đã báo cáo trong thử nghiệm VideoIntel (trung bình và bảng chi tiết).

**Tổng quan trung bình**

| Chỉ số | Giá trị |
|--------|---------|
| Tỷ lệ thời lượng summary / gốc | ~**21,5%** |
| Giảm dung lượng file (so với gốc) | ~**76,8%** |
| Tốc độ Whisper (gốc / tóm tắt) | ~**4,4×** nhanh hơn trên tóm tắt |
| Coverage transcript trung bình (gốc / tóm tắt) | ~**93,8%** / ~**94,2%** |
| Tỷ lệ số đoạn còn lại (summ / orig) | ~**25,6%** |
| Chênh CLIP top-1 trung bình (3 truy vấn, summ − orig) | **+0,0015** (gần như không đổi) |
| Q&A định tính (chấp nhận được) | Gốc **8/10**, tóm tắt **5/10** |

**Nén thời lượng và dung lượng (mỗi video)**

| Video | Orig (s) | Summ (s) | Ratio | Orig (MB) | Summ (MB) | Giảm size | Ghi chú |
|-------|----------|----------|-------|-----------|-----------|-----------|---------|
| video_01 | 351 | 76 | 21,7% | 17,22 | 3,8 | 77,9% | Ổn định |
| video_02 | 1198 | 254 | 21,2% | 75,29 | 15,9 | 78,9% | Ổn định |
| video_03 | 480 | 104 | 21,7% | 25,86 | 4,75 | 81,6% | Ổn định |
| video_04 | 488 | 111 | 22,7% | 21,48 | 8,76 | 59,2% | Giảm dung lượng thấp hơn dù tỷ lệ thời gian tương tự |
| video_05 | 1555 | 342 | 22,0% | 73,16 | 16,06 | 78,0% | Ổn định |
| video_06 | 927 | 193 | 20,8% | 48,01 | 11,15 | 76,8% | Ổn định |
| video_07 | 1359 | 314 | 23,1% | 71,41 | 16,05 | 77,5% | Ổn định |
| video_08 | 831 | 177 | 21,3% | 49,93 | 10,43 | 79,1% | Ổn định |
| video_09 | 1054 | 210 | 19,9% | 60,34 | 11,83 | 80,4% | Ổn định |
| video_10 | 1006 | 209 | 20,8% | 54,61 | 11,77 | 78,4% | Ổn định |
| **Trung bình** | **825** | **179** | **21,5%** | **50,6** | **11,1** | **76,8%** | |

**Thời gian xử lý Whisper và mã hóa CLIP (60 khung)**

| Video | Whisper gốc (s) | Whisper tóm tắt (s) | Speedup | CLIP gốc (s) | CLIP tóm tắt (s) |
|-------|-----------------|---------------------|---------|--------------|------------------|
| video_01 | 109,29 | 50,52 | 2,2× | 3,03 | 3,15 |
| video_02 | 112,15 | 25,2 | 4,5× | 3,45 | 3,32 |
| video_03 | 97,64 | 15,91 | 6,1× | 3,46 | 3,44 |
| video_04 | 207,15 | 29,73 | 7,0× | 3,16 | 3,09 |
| video_05 | 132,18 | 36,23 | 3,6× | 3,2 | 3,12 |
| video_06 | 155,04 | 44,66 | 3,5× | 3,31 | 3,21 |
| video_07 | 194,76 | 44,11 | 4,4× | 3,12 | 3,32 |
| video_08 | 117,31 | 25,34 | 4,6× | 3,07 | 2,99 |
| video_09 | 148,5 | 33,3 | 4,5× | 3,26 | 3,34 |
| video_10 | 155,26 | 38,14 | 4,1× | 3,13 | 3,13 |
| **Trung bình** | **142,9** | **34,3** | **4,4×** | **3,22** | **3,21** |

**Độ phủ và số đoạn transcript**

| Video | Cov gốc | Cov tóm tắt | Seg gốc | Seg tóm tắt | % đoạn còn lại |
|-------|---------|-------------|---------|-------------|----------------|
| video_01 | 98,4% | 100% | 103 | 31 | 30,1% |
| video_02 | 93,7% | 99,4% | 478 | 71 | 14,9% |
| video_03 | 93,1% | 93% | 89 | 19 | 21,3% |
| video_04 | 91,5% | 97,7% | 84 | 42 | 50,0% |
| video_05 | 96,1% | 93,5% | 320 | 138 | 43,1% |
| video_06 | 94,1% | 92,3% | 202 | 51 | 25,2% |
| video_07 | 88,7% | 94% | 400 | 81 | 20,3% |
| video_08 | 93,4% | 85,3% | 164 | 33 | 20,1% |
| video_09 | 92,5% | 89,7% | 378 | 36 | 9,5% |
| video_10 | 96,8% | 97,2% | 214 | 46 | 21,5% |
| **TB** | **93,8%** | **94,2%** | **243** | **55** | **25,6%** |

**CLIP top-1 theo ba truy vấn (Person Speaking / Important Event / Diagram or Chart)**

Trung bình toàn bộ video: điểm trung bình trên PS là **0,2509 (gốc)** vs **0,2524 (tóm tắt)**; IE **0,2354** vs **0,2374**; DC **0,2274** vs **0,2271**; chênh trung bình (summ − orig) trên ba truy vấn và mười video: **+0,0015** — **tương đương thống kê**.

### 9.7. Kết quả định tính Q&A (minh họa)

Với cùng một câu hỏi mô tả tổng thể, các trường hợp tiêu biểu: **video_01** tóm tắt **sai chủ đề nghiêm trọng** so với gốc; **video_02, 04, 05, 10** **khớp** chủ đề; **video_03, 06, 07, 08, 09** **một phần đúng**, thiếu chi tiết hoặc có chi tiết **không có trong gốc** (ảo giác / lệch ngữ cảnh), trong đó **video_09** mất ~90% số đoạn transcript — phù hợp mức **thiếu thông tin** nặng.

### 9.8. Hạn chế và khuyến nghị (ứng dụng)

- Cỡ mẫu 10 video; một LLM và một dạng câu hỏi — **không** khái quát hết miền ngôn ngữ và nội dung.  
- Điểm CLIP top-1 **không** thay thế đánh giá relevance có nhãn.  
- Khuyến nghị: đặt **ngưỡng tối thiểu** số đoạn transcript trước khi tin cậy Q&A; kiểm tra **nhất quán chủ đề** giữa transcript gốc và tóm tắt; xác minh audio khi coverage giảm bất thường.

---

## 10. Kết luận

Báo cáo đã trình bày **mô hình đa phương thức BiLSTM có Temporal Attention** với **đặc trưng 2432 chiều**, huấn luyện có giám sát và suy luận theo **\(r=0{,}35\)**, kèm **kết quả thực nghiệm** và **ablation** (mục 8.1) cùng **bảng so sánh văn bản nghiên cứu** (mục 8.2). Trên nhánh ứng dụng, **VideoIntel** cho thấy **tìm kiếm CLIP** trên video tóm tắt **gần như không suy giảm** ở mức metric đã định nghĩa, trong khi **hỏi–đáp** phụ thuộc mạnh vào **mật độ và độ đúng chủ đề của transcript** — phản ánh đúng vai trò của tóm tắt như **nén có mất mát** và nhu cầu **kiểm chứng downstream** trước khi triển khai thực tế.

---

## 11. Tài liệu tham khảo

Danh mục dưới đây gồm: **bộ dữ liệu** và **phương pháp tóm tắt video** (mục 8), **mô hình nền và công cụ** trong pipeline (ResNet, CLIP, Whisper, Sentence-BERT, v.v.), **phần mềm** phục vụ ứng dụng VideoIntel, các **bài khảo sát và nghiên cứu đa phương thức** thường dùng để đặt bài toán trong bối cảnh rộng hơn, **kiến trúc nhận diện video** tham chiếu cho hướng mở rộng downstream, và **cơ sở lý thuyết thuật toán tối ưu** (Adam / AdamW). Các mục được nhóm theo chủ đề; phần [19] trở đi mở rộng danh mục nền tảng ngoài các trích dẫn trực tiếp trong bảng so sánh mục 8.2.

### 11.1. Bộ dữ liệu và đánh giá tóm tắt video

1. E. Apostolidis, K. Mekkas, A. I. Patras, I. Kompatsiaris, “PGL-SUM: A Novel Pointer-Generator Network for Extractive Video Summarization,” *Proc. ACM Int. Conf. on Multimedia Retrieval (ICMR)* / bản ISM 2021 (preprint). URL: https://www.iti.gr/~bmezaris/publications/ism2021a_preprint.pdf  
2. M. Gygli, H. Grabner, H. Riemenschneider, L. Van Gool, “Creating summaries from user videos,” in *Computer Vision — ECCV 2014*, Lecture Notes in Computer Science, vol. 8695, Springer, 2014, pp. 505–520. (Tập **SumMe**.)  
3. Y. Song, J. Vallmitjana, A. Stent, A. Jaimes, “TVSum: Summarizing web videos using titles,” in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2015, pp. 5179–5187. (Tập **TVSum**.)

### 11.2. Phương pháp tóm tắt video (so sánh mục 8.2)

4. J. Fajtl, H. S. Sokeh, V. Argyriou, D. Monekosso, P. Remagnino, “Summarizing Videos with Attention,” *Asian Conference on Computer Vision (ACCV)*, 2018 (thường gọi **VASNet**). Repo tham khảo: https://github.com/azhar0100/VASNet  
5. S. A. Ghauri, F. S. Khan, S. W. Zamir, J. Hayat, M. Shah, “MSVA: Multi-Stage Aggregated Transformer for Video Summarization,” *arXiv preprint* arXiv:2104.11530, 2021. URL: https://arxiv.org/pdf/2104.11530.pdf  
6. B. Mahasseni, M. Lam, S. Todorovic, “Unsupervised Video Summarization with Adversarial LSTM Networks,” *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017 (**SUM-GAN**). URL: https://web.engr.oregonstate.edu/~sinisa/research/publications/cvpr17_summarization.pdf  
7. K. Zhang, W.-L. Chao, F. Sha, K. Grauman, “Video Summarization with Long Short-Term Memory,” *European Conference on Computer Vision (ECCV)*, 2016. Trong cùng bài đề xuất hai biến thể **vsLSTM** và **dppLSTM** cho tóm tắt video.  
8. K. Zhang, K. Grauman, F. Sha, “Retrospective Encoders for Video Summarization,” *AAAI Conference on Artificial Intelligence*, 2019 (**Cycle-SUM**). URL: https://ojs.aaai.org/index.php/AAAI/article/view/4948/4821  
9. Các bài được gom nhãn **SUM-GDA** (khoảng 2020) trong một số bảng so sánh literature — cần đối chiếu paper gốc và protocol; có thể khác các dòng khác trong bảng 8.2.  
10. W. Zhu, J. Lu, J. Li, J. Zhou, “DSNet: A Flexible Detect-to-Summarize Network for Video Summarization,” *IEEE Transactions on Image Processing (TIP)*, 2021 (phiên bản preprint/OpenReview; có nhánh anchor-free). Repo: https://github.com/li-plus/DSNet  
11. Y. Zhu, J. Li, L. Zhang, et al., “MHSCNet: Multi-Head Self-Attention Based Convolutional Network for Video Summarization,” *arXiv preprint* arXiv:2204.08352, 2022. URL: https://arxiv.org/pdf/2204.08352  

*(Khi so sánh số F-score giữa các phương pháp trong mục 8.2, cần đối chiếu **cùng protocol** như đã lưu ý.)*

### 11.3. Mạng nền, đa phương thức và học sâu chung

12. K. He, X. Zhang, S. Ren, J. Sun, “Deep Residual Learning for Image Recognition,” *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016. (**ResNet-50** làm backbone trích đặc trưng 2048 chiều.)  
13. S. Hochreiter, J. Schmidhuber, “Long Short-Term Memory,” *Neural Computation*, 9(8), 1735–1780, 1997.  
14. A. Radford, J. W. Kim, C. Hallacy, et al., “Learning Transferable Visual Models From Natural Language Supervision,” *International Conference on Machine Learning (PMLR)*, 2021 (**CLIP**). Triển khai mã nguồn mở: OpenCLIP (ví dụ ViT-B/32, trọng số OpenAI).  
15. N. Reimers, I. Gurevych, “Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks,” *Conference on Empirical Methods in Natural Language Processing (EMNLP)*, 2019. (**SentenceBERT / SBERT**, chiều 384 trong báo cáo.)  

### 11.4. Nhận dạng tiếng nói, LLM cục bộ và công cụ hệ thống

16. A. Radford et al., “Robust Speech Recognition via Large-Scale Weak Supervision,” *Proc. Int. Conf. on Machine Learning (ICML)*, 2023 (**Whisper**).  
17. Ollama — runtime chạy mô hình ngôn ngữ cục bộ qua API HTTP (ví dụ Llama 3.2). Trang chủ: https://ollama.com  

### 11.5. FFmpeg

18. FFmpeg Developers, *FFmpeg: A complete, cross-platform solution to record, convert and stream audio and video*. https://ffmpeg.org  

### 11.6. Khảo sát, tổng quan và đa phương thức

Các mục [19]–[24] là bài **review**, **survey** và nghiên cứu **đa phương thức** thường được trích khi phân tích bối cảnh và xu hướng tóm tắt video.

19. H. B. Haq, M. Asif, M. B. Ahmad, R. Ashraf, T. Mahmood, “Video summarization techniques: A review,” *Mathematical Problems in Engineering*, vol. 2021, pp. 1–17, 2021.  
20. A. G. Money, H. Agius, “A survey on video summarization techniques,” *International Journal of Computer Applications*, vol. 118, no. 11, pp. 25–31, 2015.  
21. T. Psallidas, P. Koromilas, T. Giannakopoulos, E. Spyrou, “Multimodal summarization of user-generated videos,” *Applied Sciences*, vol. 11, no. 11, p. 5260, 2021.  
22. S. Hu, Z. Liu, J. Liu, Z. Guo, “Video summarization based on feature fusion and data augmentation,” *Computers*, vol. 12, no. 9, p. 186, 2023.  
23. P. Kadam *et al.*, “Recent challenges and opportunities in video summarization with machine learning algorithms,” *IEEE Access*, vol. 10, pp. 122762–122785, 2022.  
24. E. Apostolidis, E. Adamantidou, A. I. Metsai, V. Mezaris, I. Patras, “Video summarization using deep neural networks: A survey,” *Proceedings of the IEEE*, vol. 109, no. 11, pp. 1838–1863, 2021. *(Khác với [1] — đây là bài khảo sát tổng quan; [1] là PGL-SUM / mạng pointer-generator.)*

### 11.7. Kiến trúc nhận diện video (tham chiếu cho tác vụ downstream)

Các công trình sau là **baseline kiến trúc** phổ biến khi đánh giá mô hình nhận diện trên video; có thể dùng làm tham chiếu lý thuyết khi so sánh huấn luyện trên **video gốc** và trên **video đã rút gọn**.

25. G. Bertasius, H. Wang, L. Torresani, “Is Space-Time Attention All You Need for Video Understanding?,” *Proc. Int. Conf. Machine Learning (ICML)*, 2021 (**TimeSformer**).  
26. Z. Liu, J. Ning, Y. Cao, Y. Wei, Z. Zhang, S. Lin, H. Hu, “Video Swin Transformer,” *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022.  
27. J. Carreira, A. Zisserman, “Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset,” *IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017 (**I3D** / Inflated 3D ConvNet).

### 11.8. Thuật toán tối ưu Adam và AdamW

28. D. P. Kingma, J. Ba, “Adam: A Method for Stochastic Optimization,” *International Conference on Learning Representations (ICLR)*, 2015.  
29. I. Loshchilov, F. Hutter, “Decoupled Weight Decay Regularization,” *International Conference on Learning Representations (ICLR)*, 2019 (**AdamW**).

### 11.9. Liên kết nội dung giữa các mục trích dẫn

- Các mục **[7]**, **[14]**, **[15]**, **[16]** lần lượt tương ứng **BiLSTM / LSTM cho tóm tắt video** (Zhang et al., ECCV 2016), **CLIP**, **Sentence-BERT**, **Whisper** — các công trình nền tảng được nhắc lại nhiều lần trong báo cáo vì vai trò trực tiếp trong pipeline.  
- Thuật toán tối ưu **AdamW** dùng trong huấn luyện có cơ sở lý thuyết tại **[28]** và **[29]**.

---

## Phụ lục: Thí nghiệm ablation

Các biến thể **BiLSTM chỉ hình**, **full đa phương thức**, **không attention**, **UniLSTM** đều đã có **số liệu** ở mục 8.1. Các hướng mở rộng khác (domain mới, downstream như phân loại / captioning trên video tóm tắt) có thể phát triển tiếp ngoài phạm vi báo cáo này.
