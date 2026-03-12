## Đề tài: Tóm tắt video tự động bằng BiLSTM đa phương thức (visual + audio)

---

### 1. Tổng quan

#### 1.1. Video summarization là gì?

**Video summarization** (tóm tắt video) là bài toán tạo ra một phiên bản rút gọn của video gốc nhưng vẫn giữ được các nội dung/diễn biến chính [1][2]. Tóm tắt có thể ở dạng:

- **Static summary**: tập các keyframes (khung hình tiêu biểu).
- **Dynamic summary**: một video ngắn đã được cắt ghép lại từ các đoạn quan trọng, có cả hình ảnh và âm thanh.

Mục tiêu là **giảm thời lượng xem** nhưng **không làm mất mạch nội dung** và vẫn truyền tải được ý chính cho người xem.

#### 1.2. Ví dụ trong đời thường

- **Ví dụ đời thường (gần gũi)**  
  Người dùng muốn xem một video lecture 1 tiếng trên YouTube nhưng chỉ có 10 phút. Một hệ thống video summarization có thể:
  - Tự động xác định những đoạn giảng giải trọng tâm (nơi giảng viên giới thiệu khái niệm, ví dụ minh hoạ, kết luận).
  - Cắt ghép thành một video dài khoảng 6–10 phút, giúp người xem nắm được 70–80% nội dung quan trọng mà không cần xem toàn bộ 60 phút.

- **Ví dụ dưới góc nhìn giảm chiều dữ liệu (data reduction)**  
  Trong bài toán **video classification** (phân loại video), mô hình thường phải xử lý hàng nghìn frame cho mỗi video → chi phí tính toán rất lớn. Nếu trước khi đưa vào mô hình phân loại, ta **tóm tắt video** để lấy ra:
  - Chỉ khoảng **15% số frame** quan trọng, hoặc
  - Chỉ các đoạn/shot tiêu biểu (dynamic summary),

  thì bài toán phân loại về bản chất đang làm việc trên một **đại diện có số chiều giảm** (ít frame hơn, ít thông tin thừa hơn). Điều này:
  - Giảm chi phí tính toán và dung lượng lưu trữ.
  - Tập trung vào những phần chứa thông tin phân biệt lớp, giúp mô hình học hiệu quả hơn.

---

### 2. Ý nghĩa

#### 2.1. Ý nghĩa thực tế

Từ hai ví dụ trên, có thể rút ra các ý nghĩa thực tế [5]:

- **Tiết kiệm thời gian cho người dùng**: Người dùng không cần xem toàn bộ video dài, chỉ xem bản tóm tắt nhưng vẫn nắm được nội dung chính.
- **Hỗ trợ hệ thống khuyến nghị và tìm kiếm**: Tóm tắt video giúp:
  - Tạo thumbnail/keyframe ý nghĩa hơn.
  - Cho phép người dùng xem nhanh bản preview trước khi quyết định xem đầy đủ.
- **Ứng dụng trong giáo dục và tin tức**:
  - Tự động tạo video highlight cho lecture, MOOC.
  - Tạo bản tin tóm tắt từ một chương trình news dài.

#### 2.2. Ý nghĩa khoa học

Dưới góc độ khoa học dữ liệu:

- **Giảm chiều nhưng vẫn giữ thông tin cốt lõi**: Video summarization là một dạng **data reduction có cấu trúc**, trong đó:
  - Không chỉ giảm số lượng frame, mà còn chọn đúng những frame/đoạn có **thông tin cao** về nội dung.
- **Tiền xử lý cho các mô hình downstream**:
  - Với một tập video lớn, việc tóm tắt trước khi huấn luyện các mô hình khác (classification, retrieval, captioning, QA) giúp:
    - Giảm chi phí training/inference.
    - Tập trung vào phần giàu thông tin, tiềm năng giúp mô hình đạt hiệu năng tốt hơn trên cùng tài nguyên.

---

### 3. Bối cảnh nghiên cứu

Các hướng tiếp cận video summarization hiện nay khá đa dạng:

- **Truyền thống**: dựa trên heuristic, clustering, detection shot, rule-based.
- **Deep learning**:
  - Mô hình **sequence labeling** (LSTM/BiLSTM) dự đoán độ quan trọng từng frame, như trong các work baseline trên SumMe, TVSum [9].
  - Các mô hình dựa trên **attention/Transformer**, self-supervised, reinforcement learning.
  - Xu hướng **multimodal**: kết hợp hình ảnh, âm thanh, transcript, caption, metadata [3][4].

Các survey gần đây về video summarization [7] tổng hợp rằng:

- Bài toán vẫn đang phát triển mạnh nhờ:
  - Sự bùng nổ của video online.
  - Nhu cầu tóm tắt cho nhiều domain (news, lecture, surveillance, sports).
- Cơ hội phát triển:
  - **Khai thác tốt hơn thông tin văn bản/audio** (speech, subtitle, OCR).
  - **Mô hình hoá chuỗi dài** tốt hơn (Transformer cho video dài, hierarchical modeling).
  - **Học không giám sát / bán giám sát**, giảm phụ thuộc vào label tốn kém [7].

**Hướng nhóm tập trung** trong bối cảnh này:

- Tập trung vào **video summarization đa phương thức (visual + audio)**, supervised trên SumMe & TVSum.
- Sử dụng kiến trúc **BiLSTM + Temporal Attention**:
  - Giữ ưu điểm của LSTM cho chuỗi dài.
  - Kết hợp được thông tin hình ảnh + âm thanh/lời nói.
- Cải tiến:
  - Cách chọn keyshot và ghép video (dynamic summary).
  - Pipeline audio (Whisper + SentenceBERT) để khai thác nội dung ngữ nghĩa.

---

### 4. Phát biểu bài toán

#### 4.1. Đầu vào (Input)

- Video thô $V$ dạng `.mp4`, `.avi`, …, độ dài tùy ý.
- Trong quá trình huấn luyện:
  - Với dataset SumMe/TVSum: có thêm label frame-level $[0,1]$ được gán bởi 15–20 annotators.

#### 4.2. Đầu ra (Output)

- **Static summary**:
  - Tập hợp các **keyframes** tiêu biểu đại diện cho video.
- **Dynamic summary**:
  - Một video ngắn $V_{sum}$ (khoảng **15% độ dài** so với video gốc) được cắt ghép từ các đoạn quan trọng, có cả audio.

#### 4.3. Hướng triển khai của nhóm

- **Nhóm tập trung chính vào: Dynamic summary**  
  Lý do:
  - Gần với nhu cầu thực tế hơn (người dùng muốn xem một bản video rút gọn có cả tiếng, không chỉ ảnh tĩnh).
  - Cho phép **đánh giá chất lượng tóm tắt trực quan** qua trải nghiệm xem.
  - Tận dụng tốt pipeline **ffmpeg + xử lý audio** mà project đã xây dựng.
- **Static summary** vẫn được hỗ trợ:
  - Từ các keyframes / keyshot đã chọn, ta có thể xuất ra thêm dạng static nếu cần.

---

### 5. Phương pháp: Supervised Learning, Dataset, Thang đo đánh giá

#### 5.1. Học có giám sát (Supervised Learning)

- Mô hình học để **xấp xỉ đánh giá của con người**:
  - Đầu vào: chuỗi feature $[T, D]$ (visual + audio) của từng frame.
  - Đầu ra: score $\hat{y}_t \in [0,1]$ cho mỗi frame $t$.
  - Ground truth: $y_t \in [0,1]$ là điểm quan trọng trung bình từ annotators.
- Loss function: thường là **MSE/L1 loss** giữa $\hat{y}_t$ và $y_t$, hoặc các biến thể tối ưu F-score.

#### 5.2. Dataset sử dụng

- **SumMe** [8]:
  - Video ngắn, đa dạng chủ đề.
  - Annotators đánh dấu đoạn quan trọng, sau đó chuyển thành frame-level importance score.
- **TVSum** [10]:
  - Video mang tính "TV program" hơn (news, documentary, vlog).
  - Tương tự, có khoảng 15–20 annotators, score trung bình tạo thành ground truth.

#### 5.3. Thang đo đánh giá

Các metric chính:

- **F-score** (F-measure):
  - Đo sự trùng khớp giữa tóm tắt sinh ra và tóm tắt "chuẩn" từ annotators (dựa trên temporal overlap).
- **Precision / Recall**:
  - Precision: tỷ lệ nội dung trong summary là "thực sự quan trọng".
  - Recall: tỷ lệ nội dung quan trọng trong video gốc được summary giữ lại.
- **Temporal Overlap**:
  - Đo mức độ trùng phủ thời gian giữa các đoạn summary và đoạn ground truth.

---

### 6. Kỹ thuật BiLSTM

#### 6.1. Mô tả kỹ thuật

- **BiLSTM (Bidirectional Long Short-Term Memory)** [9]:
  - Là kiến trúc LSTM chạy theo **hai chiều thời gian** (forward & backward).
  - Với mỗi frame $t$, hidden state được tính từ:
    - Thông tin trước đó (quá khứ).
    - Thông tin sau đó (tương lai) trong chuỗi.
- Trong bài toán video summarization:
  - Chuỗi input là $[x_1, x_2, \dots, x_T]$, mỗi $x_t \in \mathbb{R}^{1408}$ (visual + audio).
  - BiLSTM cho ra biểu diễn ẩn $[h_1, \dots, h_T]$, mỗi $h_t \in \mathbb{R}^{2H}$.
  - Sau đó áp dụng **Temporal Attention**:
    - Học trọng số chú ý $\alpha_t$ cho từng $h_t$.
    - Tập trung mạnh hơn vào những frame quan trọng để dự đoán score.

#### 6.2. Lý do chọn BiLSTM

- Khả năng **nắm bắt ngữ cảnh dài**:
  - Frame quan trọng thường phụ thuộc vào ngữ cảnh trước–sau, không thể quyết định chỉ bằng thông tin cục bộ.
- BiLSTM tận dụng được **cả quá khứ và tương lai** của timeline video [9].
- Kết hợp tốt với **attention** để:
  - Tập trung vào những vùng thời gian quan trọng.
  - Giảm ảnh hưởng của nhiễu và đoạn ít quan trọng.

#### 6.3. Thuật toán huấn luyện mô hình

- **Bước 1 – Tiền xử lý dữ liệu & trích xuất đặc trưng**
  - **B1.1 – Chuẩn bị video và annotation**
    - Tải dataset SumMe [8] / TVSum [10].
    - Chuyển annotation gốc (segment-level importance) về **frame-level score** $y_t \in [0,1]$ cho từng frame.
  - **B1.2 – Trích xuất visual features**
    - Dùng `FrameSampler` lấy frame với tần số 2 fps từ mỗi video → chuỗi frame $\{f_t\}_{t=1}^T$.
    - Dùng CNN (GoogLeNet/ResNet) trích xuất feature cho mỗi frame:
      $$v_t = \text{CNN}(f_t) \in \mathbb{R}^{1024}$$
    - Lưu thành file `{video_id}.npy` shape `[T, 1024]`.
  - **B1.3 – Trích xuất audio features**
    - Dùng `ffmpeg` tách audio từ video thành `.wav`.
    - Dùng **Whisper** [6] để:
      - Nhận transcript từng câu nói.
      - Lấy timestamp bắt đầu–kết thúc cho mỗi câu.
    - Dùng **SentenceBERT** [11] để mã hoá mỗi câu:
      $$a^{\text{text}}_k = \text{SentenceBERT}(\text{câu}_k) \in \mathbb{R}^{384}$$
    - Căn chỉnh các vector này theo time-line frame:
      - Với mỗi frame $t$ (biết timestamp), gán $a_t = a^{\text{text}}_k$ nếu frame nằm trong đoạn thời gian của câu $k$; ngược lại $a_t = \mathbf{0}$.
    - Lưu thành `{video_id}_audio.npy` shape `[T, 384]`.

- **Bước 2 – Tạo chuỗi input và batch training**
  - **B2.1 – Kết hợp visual + audio**
    - Với mỗi frame $t$, ghép:
      $$x_t = [v_t ; a_t] \in \mathbb{R}^{1408}$$
    - Kết quả: chuỗi feature **multimodal** $\{x_t\}_{t=1}^T$ cho mỗi video.
  - **B2.2 – Chia batch & xử lý độ dài**
    - Nếu video quá dài, cắt/chunk thành những đoạn có độ dài tối đa `max_seq_len` (ví dụ 960 frame).
    - Tạo DataLoader:
      - Shuffle các video/segment.
      - Pad sequence trong cùng batch nếu cần, mask loss cho phần padding.

- **Bước 3 – Forward qua BiLSTM + Attention**
  - **B3.1 – BiLSTM encoding**
    - Đưa chuỗi $(x_1,\dots,x_T)$ vào BiLSTM:
      $$(h_1,\dots,h_T) = \text{BiLSTM}(x_1,\dots,x_T),\quad h_t \in \mathbb{R}^{2H}$$
  - **B3.2 – Temporal Attention**
    - Tính attention score cho từng $h_t$:
      $$e_t = \text{tanh}(W_h h_t + b_h), \quad \alpha_t = \text{softmax}(w^\top e_t)$$
    - Attention có thể được dùng:
      - Để tạo representation global của sequence.
      - Hoặc để điều chỉnh trọng số khi dự đoán score frame.
  - **B3.3 – Dự đoán score frame**
    - Với mỗi frame, qua một MLP/linear layer:
      $$\hat{y}_t = \sigma(W_o h_t + b_o),\quad \hat{y}_t \in [0,1]$$
    - Thu được chuỗi dự đoán $\{\hat{y}_t\}_{t=1}^T$.

- **Bước 4 – Tính loss & tối ưu**
  - **B4.1 – Tính loss giám sát**
    - So sánh với ground truth $y_t$:
      $$\mathcal{L} = \frac{1}{T}\sum_{t=1}^{T} \lVert \hat{y}_t - y_t \rVert_1 \quad \text{(L1 loss, hoặc MSE)}$$
  - **B4.2 – Backprop và update tham số**
    - Tính gradient của loss theo toàn bộ tham số trong:
      - BiLSTM (các layer forward & backward).
      - Attention.
      - Output layer.
    - Dùng optimizer (Adam) với learning rate từ `config.yaml` (vd 0.001) để cập nhật.
  - **B4.3 – Validation & early stopping**
    - Sau mỗi epoch:
      - Tính loss trên tập validation (val_loss).
      - Nếu val_loss tốt hơn trước đó → lưu `checkpoints/best.pt`.
      - Nếu val_loss không cải thiện sau `early_stopping_patience` epoch → dừng sớm (early stopping).

#### 6.4. Thuật toán inference & sinh video tóm tắt

- **I1 – Trích xuất feature cho video mới**
  - Dùng lại `frame_sampler`, `cnn_extractor`, `audio_extractor` như trong huấn luyện để tạo chuỗi $\{x_t\}_{t=1}^T$.

- **I2 – Dự đoán importance score**
  - Đưa chuỗi $\{x_t\}$ vào BiLSTM + Attention đã train (load từ `best.pt`) để lấy $\hat{y}_t$ cho từng frame.

- **I3 – Hậu xử lý score**
  - Áp dụng **Gaussian smoothing** lên chuỗi $\hat{y}_t$ để giảm nhiễu.
  - Chuẩn hoá score nếu cần, đảm bảo phân bố mượt cho bước chọn keyshot.

- **I4 – Chọn keyshot & keyframes**
  - Cắt video thành các **shot** liên tục.
  - Tính tổng hoặc trung bình score cho từng shot.
  - Chọn các shot có tổng score cao nhất sao cho độ dài tổng **≈ summary_ratio × độ dài video**.
  - Đảm bảo:
    - Các shot được chọn **đa dạng, trải đều** toàn timeline.
    - Có **minimum gap** giữa các keyframes để tránh trùng lặp.

- **I5 – Xuất dynamic summary**
  - Dùng `ffmpeg` cắt/ghép lại các segment đã chọn (mỗi segment có mở rộng ±1.5s để giữ ngữ cảnh).
  - Ghép audio track tương ứng → tạo video tóm tắt có âm thanh hoàn chỉnh.

---

### 7. Xử lý âm thanh

#### 7.1. Pipeline xử lý audio trong huấn luyện

- **Bước 1 – Tách audio**
  - Dùng `ffmpeg` tách audio từ video gốc thành file `.wav`.

- **Bước 2 – Transcribe với Whisper** [6]
  - Dùng **Whisper** (openai-whisper) để chuyển audio thành:
    - Câu nói (text).
    - Timestamp bắt đầu–kết thúc cho từng câu/segment.

- **Bước 3 – Mã hoá text bằng SentenceBERT** [11]
  - Mỗi câu transcript → một vector embedding `[384]` thông qua SentenceBERT.
  - Các embedding này mang thông tin **ngữ nghĩa của nội dung lời nói**.

- **Bước 4 – Căn chỉnh với frame**
  - Với mỗi frame (biết timestamp), gán:
    - Embedding của câu đang "bao phủ" frame đó nếu có lời thoại.
    - Hoặc vector zero nếu frame đó im lặng/không có speech.
  - Kết quả: chuỗi audio feature `[T, 384]`, đồng bộ với chuỗi frame `[T, 1024]`.

- **Bước 5 – Kết hợp visual + audio**
  - Concatenate: `[1024] (visual) + [384] (audio) → [1408]`.
  - Chuỗi `[T, 1408]` này là input duy nhất được đưa vào BiLSTM.

Nhờ vậy, mô hình có thể:

- Nhận biết được **nội dung lời nói**, không chỉ hình ảnh.
- Ưu tiên những đoạn có nhiều thông tin ngữ nghĩa (giải thích, kết luận, tiêu đề slide, v.v.).

---

### 8. Kết quả hiện tại

**Kết quả định lượng với checkpoint:**

Hệ thống được đánh giá bằng các thang đo ở mục 5.3 (precision, recall, F-score và temporal overlap) Checkpoint hiện tại là mô hình **BiLSTM đa phương thức (visual + audio)** với **input_dim = 1408** (1024 visual + 384 audio), 2 lớp LSTM hidden size 256, dùng Temporal Attention, được huấn luyện trên **75 video** với early stopping tại epoch 15 và mô hình đạt:

- `Precision ≈ 0.4674`
- `Recall ≈ 0.4674`
- `F-score ≈ 0.4674`
- `Temporal overlap ≈ 0.4222`

Hiện tại các chỉ số này **chưa được so sánh trực tiếp** với kết quả từ các nghiên cứu/baseline khác trên SumMe/TVSum và sẽ được bổ sung sau.

**Kết quả định tính về chất lượng summary:**

Hệ thống đã được thử nghiệm trên một số video thực tế với độ dài khoảng **10 phút**, cho ra dynamic summary dài khoảng **2–3 phút** (tương ứng giữ lại ~20–30% độ dài). Các đoạn được chọn tập trung vào vùng có hoạt động chính, chuyển cảnh quan trọng hoặc lời thoại chứa nội dung cốt lõi. Video tóm tắt ngắn gọn, mạch lạc hơn nhờ shot-based selection, Gaussian smoothing và mở rộng ngữ cảnh ±1.5 giây quanh keyframe. Về hình ảnh, các đoạn được ghép lại tự nhiên, ít bị giật cục, khung hình thể hiện đúng nội dung nổi bật. Về âm thanh, nhờ giữ nguyên audio track qua `ffmpeg`, chất lượng âm thanh chấp nhận được, không bị mất tiếng đột ngột, trải nghiệm xem tương đối trôi chảy.

**So sánh với các nghiên cứu khác:**

Ở thời điểm hiện tại, nhóm **chưa tiến hành so sánh định lượng trực tiếp** với các phương pháp/benchmark khác trên SumMe/TVSum. Đây sẽ là bước tiếp theo quan trọng: đối chiếu các chỉ số đã đo được với kết quả trong các bài báo/survey [7][9] để đánh giá vị trí của mô hình đề xuất, từ đó đề ra hướng cải tiến phù hợp.

---

### 9. Kế hoạch phát triển tiếp theo

#### 9.1. Mở rộng test trên nhiều domain (news, education, v.v.)

- **Bước 1 – Thu thập video theo domain**
  - Tập video **news** (bản tin thời sự, talk show).
  - Tập video **education** (lecture, tutorial, MOOC).
- **Bước 2 – Chạy hệ thống tóm tắt hiện tại**
  - Dùng mô hình BiLSTM multimodal để sinh **summary** cho mỗi video:
    - Dynamic summary (video rút gọn ~15%).
    - Optionally lưu lại index frame/shot đã được chọn.

#### 9.2. Xây dựng dataset mới sau khi giảm chiều

Ý tưởng:

- Sau khi đã có video summary, ta coi đó là **phiên bản giảm chiều** của dữ liệu ban đầu.
- Từ các summary này, xây dựng một dataset mới cho một bài toán khác, ví dụ:
  - **Video classification** (phân loại chủ đề video).
  - Hoặc **video-to-text** (captioning/QA) trên video đã được rút gọn.

#### 9.3. Gợi ý mô hình downstream phù hợp

Một số mô hình phù hợp để huấn luyện trên video đã tóm tắt:

- **TimeSformer (Transformer cho video)**:
  - Là mô hình Vision Transformer mở rộng theo cả chiều không gian và thời gian.
  - Làm việc tốt với chuỗi frame có độ dài vừa phải → rất phù hợp khi video đã được tóm tắt (ít frame hơn, thông tin "đặc" hơn).
- **Video Swin Transformer**:
  - Biến thể Swin cho video, sử dụng cửa sổ trượt (shifted windows) theo không gian–thời gian.
- **I3D / 3D CNN** (Inception 3D):
  - Nếu muốn một baseline đơn giản hơn, có thể dùng I3D cho phân loại video trên summary.

Hướng đề xuất:

- **Bước 1**: Dùng hệ thống summarization hiện tại để biến tập video gốc → tập video summary (giảm ~85% thời lượng).
- **Bước 2**: Gán nhãn cho video summary theo bài toán mới (ví dụ: chủ đề news, loại lecture,…).
- **Bước 3**: Train một mô hình như **TimeSformer** trên tập video summary:
  - So sánh:
    - Train trên video gốc (không tóm tắt).
    - Train trên video summary (đã giảm chiều).
  - Đánh giá:
    - Độ chính xác phân loại.
    - Thời gian huấn luyện, tài nguyên GPU.
- **Mục tiêu kiểm chứng**:
  - Video summarization **thực sự giúp giảm số chiều dữ liệu** nhưng:
    - Vẫn giữ được (hoặc cải thiện) hiệu năng bài toán downstream.
    - Giảm đáng kể chi phí tính toán.

---

### 10. Kết luận

Đề tài tập trung vào việc xây dựng **hệ thống tóm tắt video tự động** dựa trên **BiLSTM + Temporal Attention** [9] kết hợp **visual + audio** [3][4], huấn luyện có giám sát trên SumMe [8] và TVSum [10]. Hệ thống không chỉ có ý nghĩa trong việc **tiết kiệm thời gian xem** cho người dùng mà còn đóng vai trò như một bước **giảm chiều dữ liệu có cấu trúc**, hỗ trợ cho các mô hình video downstream (classification, retrieval, captioning). Trong tương lai, nhóm dự định mở rộng test trên nhiều domain (news, education), và kiểm chứng vai trò của video summarization trong việc **tăng hiệu quả huấn luyện** cho các mô hình hiện đại như TimeSformer trên tập video đã được rút gọn.

---

### 11. Tài liệu tham khảo

[1] H. B. Haq, M. Asif, M. B. Ahmad, R. Ashraf, and T. Mahmood, "Video Summarization Techniques: A Review," _Mathematical Problems in Engineering_, vol. 2021, pp. 1–17, 2021.

[2] A. G. Money and H. Agius, "A Survey on Video Summarization Techniques," _International Journal of Computer Applications_, vol. 118, no. 11, pp. 25–31, 2015.

[3] T. Psallidas, P. Koromilas, T. Giannakopoulos, and E. Spyrou, "Multimodal summarization of user-generated videos," _Applied Sciences_, vol. 11, no. 11, p. 5260, 2021.

[4] S. Hu, Z. Liu, J. Liu, and Z. Guo, "Video summarization based on feature fusion and data augmentation," _Computers_, vol. 12, no. 9, p. 186, 2023.

[5] P. Kadam, D. Vora, S. Mishra, S. Patil, K. Kotecha, A. Abraham, and L. A. Gabralla, "Recent challenges and opportunities in video summarization with machine learning algorithms," _IEEE Access_, vol. 10, pp. 122762–122785, 2022, doi: 10.1109/ACCESS.2022.3223379.

[6] A. Radford, J. W. Kim, T. Xu, G. Brockman, C. McLeavey, and I. Sutskever, "Robust speech recognition via large-scale weak supervision," in _Proc. 40th Int. Conf. Machine Learning (ICML)_, vol. 202, pp. 28492–28518, 2023.

[7] E. Apostolidis, E. Adamantidou, A. I. Metsai, V. Mezaris, and I. Patras, "Video summarization using deep neural networks: A survey," _Proc. IEEE_, vol. 109, no. 11, pp. 1838–1863, 2021, doi: 10.1109/JPROC.2021.3117472.

[8] M. Gygli, H. Grabner, H. Riemenschneider, and L. Van Gool, "Creating summaries from user videos," in _Computer Vision – ECCV 2014_, Lecture Notes in Computer Science, vol. 8695, Springer, Cham, 2014, pp. 505–520, doi: 10.1007/978-3-319-10584-0_33.

[9] K. Zhang, W.-L. Chao, F. Sha, and K. Grauman, "Video summarization with long short-term memory," in _Computer Vision – ECCV 2016_, Lecture Notes in Computer Science, vol. 9911, Springer, Cham, 2016, pp. 766–782, doi: 10.1007/978-3-319-46478-7_47.

[10] Y. Song, J. Vallmitjana, A. Stent, and A. Jaimes, "TVSum: Summarizing web videos using titles," in _Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)_, 2015, pp. 5179–5187, doi: 10.1109/CVPR.2015.7299154.

[11] N. Reimers and I. Gurevych, "Sentence-BERT: Sentence embeddings using Siamese BERT-networks," in _Proc. 2019 Conf. Empirical Methods in Natural Language Processing and 9th Int. Joint Conf. Natural Language Processing (EMNLP-IJCNLP)_, Hong Kong, China, Nov. 2019, pp. 3982–3992, doi: 10.18653/v1/D19-1410.

---

### 12. Ghi nhận hỗ trợ

Trong quá trình soạn thảo proposal và hoàn thiện nội dung mô tả thuật toán, nhóm có sử dụng sự hỗ trợ của các công cụ AI (ChatGPT của OpenAI, Cursor) để gợi ý cấu trúc trình bày, diễn đạt lại các ý tưởng kỹ thuật cho mạch lạc, và tổng hợp nội dung từ mã nguồn hiện có cùng log huấn luyện. Các quyết định thiết kế mô hình, triển khai code và đánh giá thực nghiệm vẫn do nhóm thực hiện và chịu trách nhiệm.
