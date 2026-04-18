# So sánh checkpoint baseline và `checkpoints-optimize`

## 1. So sánh độ chính xác (test set)

Hai checkpoint được so sánh trên **cùng tập test** (split theo `config.yaml`: `train_split` / `val_split` / `test_split`, cùng `seed`), với **cùng** `inference.summary_ratio` và pipeline đánh giá trong `compares-optimize.py` (Precision, Recall, F-score trên nhị phân hóa theo ratio; Temporal overlap giữa range keyshot dự đoán và GT).

| Checkpoint                      | Đường dẫn mặc định             |
| ------------------------------- | ------------------------------ |
| Baseline (trước / nhánh `main`) | `checkpoints/best.pt`          |
| Optimize                        | `checkpoints-optimize/best.pt` |

**Chạy đo và xem bảng số liệu:**

```bash
cd video-summarize
python compares-optimize.py
```

Tùy chỉnh đường dẫn:

```bash
python compares-optimize.py --baseline checkpoints/best.pt --optimize checkpoints-optimize/best.pt --config configs/config.yaml
```

Script in ra:

- Metadata train trong file `.pt` (nếu có): `epoch`, `best_val_loss`
- `input_dim` suy từ `lstm.weight_ih_l0` (baseline và optimize có thể khác nếu một bên multimodal, một bên chỉ visual — features sẽ pad/cắt cho khớp như `run_eval.py`)
- **Kiến trúc scorer** được nhận diện tự động từ shape `scorer.0.weight`: checkpoint **cũ** (attention không concat vào scorer, ví dụ `[256, 512]`) vs **optimize** (concat context, ví dụ `[256, 1024]`). Script in thêm `use_attention` / `fuse_attention_context` mỗi lần load.
- Bảng **Baseline | Optimize | Δ** cho Precision, Recall, F-score, Temporal overlap

**Kết quả đo trên tập test** (một lần chạy `compares-optimize.py`; có thể thay đổi nhẹ theo seed / dữ liệu):

| Metric | Baseline | Optimize | Δ (O − B) |
| :----- | -------: | -------: | --------: |
| Precision | 0.4767 | 0.5293 | +0.0526 |
| Recall | 0.4767 | 0.5293 | +0.0526 |
| F-score | 0.4767 | 0.5293 | +0.0526 |
| Temporal overlap | 0.2347 | 0.3709 | +0.1363 |

---

## 2. Những action optimize là gì

Các thay đổi so với pipeline / nhánh `main` (chi tiết: [`docs/optimize.md`](docs/optimize.md)):

1. **Kiến trúc BiLSTM + attention** — Context attention toàn video được **nối (concat)** với output LSTM từng frame rồi mới đưa vào `scorer` (không chỉ dùng `lstm_out[t]`). *(Nguồn lý luận: [R1], [R2], [R3] — mục 3.)*
2. **Chọn keyshot (inference)** — **Hybrid ~70% quality / ~30% diversity**: một phần keyframe từ top shot theo score, phần còn lại bổ sung coverage trên shot chưa chọn (thay cho diversity “cứng” trên toàn bộ k). *(Nguồn ý tưởng relevance + diversity: [R4], [R5] — mục 3.)*
3. **Loss huấn luyện** — Thêm **coverage/diversity loss**: chia video thành `n_segments`, phạt variance của max score theo từng đoạn (`diversity_weight`, `n_segments` trong config). *(Nguồn loss phụ / đa mục tiêu: [R6] — mục 3.)*
4. **Optimizer / scheduler** — `weight_decay` tăng lên **1e-4**; **`ReduceLROnPlateau`** theo `val_loss` (thay `StepLR` cố định); log thêm `train_bce`, `train_div`, `lr`. *(Nguồn regularization & LR: [R7], [R8] — mục 3.)*

**Lưu ý:** Checkpoint optimize dùng kiến trúc scorer mới khi `use_attention: true` — không tương thích trọng số từng phần với checkpoint cũ; cần train lại với code optimize. So sánh độ chính xác nên dùng hai file `.pt` tương ứng hai kiến trúc như script đã load `strict=True`.

---

## 3. Tài liệu tham khảo và vì sao các kỹ thuật thường hiệu quả

Phần này tóm tắt **cơ sở trong literature / tài liệu chuẩn** (không thay cho ablation trên dữ liệu của bạn). Mã **[R1]…[R8]** trùng với ghi chú ở **mục 2**.

### 3.1 BiLSTM / RNN cho tóm tắt video và attention

- **[R1] Zhang et al., “Video Summarization with Long Short-Term Memory” (ECCV 2016)** — Nền tảng dùng LSTM để học phụ thuộc thời gian và sinh điểm quan trọng từng frame; các biến thể **vsLSTM / dppLSTM** cho thấy mô hình trình tự vượt baseline không có RNN khi có đủ nhãn. [Paper (UT Austin)](https://www.cs.utexas.edu/~grauman/papers/zhang-eccv2016-lstm-summ.pdf)
- **[R2] Fajtl et al., “Summarizing Videos with Attention” (VASNet, ACCV 2018)** — Attention trên chuỗi frame giúp **trọng số hóa** các bước thời gian liên quan tới tóm tắt, cải thiện F-score trên benchmark so với nhiều baseline chỉ pooling/FC. [arXiv](https://arxiv.org/abs/1804.08221)
- **[R3] Ji et al., “Video Summarization with Attention-Based Encoder-Decoder Networks” (AVS, arXiv 2017)** — Encoder **BiLSTM** + decoder có attention (additive / multiplicative); thể hiện rõ **attention + BiLSTM** cải thiện so với SOTA trên SumMe/TVSum trong báo cáo của tác giả; tinh thần **context-aware** gần với việc dùng thêm tín hiệu toàn chuỗi khi chấm điểm từng frame (khác chi tiết kiến trúc so với repo này). [arXiv](https://arxiv.org/abs/1708.09545)

**Tóm lại:** Literature ghi nhận LSTM + attention giúp **tận dụng phụ thuộc dài hạn** và **tập trung vào đoạn salient**; việc **đưa thêm context (ở đây concat \(\mathbf{c}\)) vào head** phù hợp tinh thần “điểm quan trọng là hàm của cả clip”, không chỉ một hidden cục bộ.

### 3.2 Diversity, coverage và chọn tập con (keyshot / shot)

- **[R4] Gygli et al., “Video Summarization by Learning Submodular Mixtures of Objectives” (CVPR 2015)** — Tóm tắt được đặt như **chọn subset** cân bằng **độ thú vị / phủ nội dung / đa dạng**; submodular có tính chất “lợi ích cận biên giảm” — gần với việc không chọn mãi cùng một kiểu đoạn. [CVF open access](https://openaccess.thecvf.com/content_cvpr_2015/papers/Gygli_Video_Summarization_by_2015_CVPR_paper.pdf)
- **[R5] Lin & Bilmes, “A Class of Submodular Functions for Document Summarization” (NAACL 2011)** — Khái niệm **coverage / diversity** trong chọn câu có thể chuyển sang chọn frame/shot; nhấn mạnh mục tiêu **không trùng lặp** và **đại diện** cho toàn tài liệu. [ACL Anthology](https://aclanthology.org/N11-1024/)

**Tóm lại:** Hybrid **một phần greedy (quality) + một phần ép phủ vùng (diversity)** nằm đúng trục trade-off mà các paper trên phân tích: tránh chỉ tối đa điểm cục bộ (bỏ sót phần khác của video) hoặc chỉ chia đều (chọn cả đoạn kém quan trọng).

### 3.3 Loss phụ (coverage/diversity) như tín hiệu bổ trợ

- **[R6] Caruana, “Multitask Learning” (Machine Learning, 1997)** — Học đồng thời nhiệm vụ liên quan tạo **inductive bias**, thường **cải thiện khái quát hóa** nếu nhiệm vụ phụ phù hợp; loss phụ có thể xem như ép representation thỏa thêm ràng buộc (ở đây: phân bố điểm cao trên nhiều segment). [Bản PDF Cornell](https://www.cs.cornell.edu/~caruana/mlj97.pdf)

**Tóm lại:** \(\mathcal{L}_{\text{BCE}} + \lambda \mathcal{L}_{\text{div}}\) là dạng **tối ưu đa mục tiêu có trọng số**; hiệu quả phụ thuộc \(\lambda\) và độ “liên quan” của \(\mathcal{L}_{\text{div}}\) với mục tiêu chính (xem [R6] về *negative transfer* nếu nhiệm vụ phụ lệch).

### 3.4 Weight decay và ReduceLROnPlateau

- **[R7] Goodfellow et al., *Deep Learning*, chương về regularization (weight decay / \(L_2\))** — Phạt độ lớn trọng số giảm **overfitting**, cải thiện lỗi trên dữ liệu chưa thấy khi mô hình trước đó quá khớp tập huấn. [Sách (MIT Press, đọc trực tuyến)](https://www.deeplearningbook.org/contents/regularization.html)
- **[R8] PyTorch `ReduceLROnPlateau`** — Giảm learning rate khi metric theo dõi (ví dụ `val_loss`) **không cải thiện** sau `patience` lần: giúp bước cập nhật nhỏ hơn gần vùng hội tụ, giảm **overshoot** so với lịch giảm LR cố định không phụ thuộc loss. [Tài liệu chính thức](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html)

### 3.5 Bảng tra mã nguồn nhanh

| Mã | Nội dung | Liên kết |
| -- | -------- | -------- |
| [R1] | LSTM cho video summarization (vsLSTM / dppLSTM) | [PDF ECCV 2016](https://www.cs.utexas.edu/~grauman/papers/zhang-eccv2016-lstm-summ.pdf) |
| [R2] | Attention trên chuỗi (VASNet) | [arXiv:1804.08221](https://arxiv.org/abs/1804.08221) |
| [R3] | Ji et al., AVS — BiLSTM encoder + attention decoder | [arXiv:1708.09545](https://arxiv.org/abs/1708.09545) |
| [R4] | Submodular mixtures, relevance & diversity | [CVPR 2015 PDF](https://openaccess.thecvf.com/content_cvpr_2015/papers/Gygli_Video_Summarization_by_2015_CVPR_paper.pdf) |
| [R5] | Submodular coverage/diversity (NLP, khái niệm) | [NAACL 2011](https://aclanthology.org/N11-1024/) |
| [R6] | Multitask / tín hiệu phụ và khái quát hóa | [Caruana MLJ97 PDF](https://www.cs.cornell.edu/~caruana/mlj97.pdf) |
| [R7] | Weight decay (\(L_2\)) | [Deep Learning Book — Regularization](https://www.deeplearningbook.org/contents/regularization.html) |
| [R8] | `ReduceLROnPlateau` | [PyTorch docs](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html) |

---

## 4. Vì sao độ chính xác tăng? (Mô tả kỹ & chứng minh toán học)

Phần này giải thích **hai lớp nguyên nhân**: (A) đảm bảo **đo đúng** mô hình đã train; (B) các **thay đổi thuật toán / kiến trúc** khiến mô hình và pipeline inference thực sự tốt hơn. Cả hai đều có thể làm metric nhảy rõ — cần tách bạch khi đọc số liệu.

### 4.1 (A) Đánh giá đúng kiến trúc checkpoint — không phải “train magic”

Checkpoint **baseline** (kiến trúc cũ) có lớp đầu của `scorer` với **đầu vào 512 chiều** (chỉ vector LSTM hai chiều: \(2 \times 256\)). Checkpoint **optimize** (kiến trúc mới) có **1024 chiều** (nối vector frame \(\mathbf{h}_t\) với vector context attention \(\mathbf{c}\), mỗi vector 512 chiều).

Nếu khi eval ta **luôn** dựng mô hình kiểu mới (1024) rồi `load_state_dict` checkpoint cũ (512), PyTorch báo lỗi size mismatch — trước khi sửa, bạn không có kết quả hợp lệ. Sau khi sửa, script **suy ra** `fuse_attention_context` từ shape `scorer.0.weight` và forward đúng như lúc train. Điều này chỉ đảm bảo **so sánh công bằng và có nghĩa**; nó không tự làm baseline “giỏi hơn”, mà tránh đo sai hoặc không chạy được.

### 4.2 (B) Metric đang được tối ưu hóa trong eval (định nghĩa chuẩn)

Trên mỗi video, với độ dài \(L\) frame và tỷ lệ tóm tắt \(r\), cả **dự đoán** và **GT** đều được nhị phân hóa cùng cách: chọn \(k = \max(1, \lfloor L \cdot r \rceil)\) frame có score cao nhất gán nhãn 1 (xem `scores_to_binary` trong `src/evaluation/metrics.py`).

Với mask nhị phân \(\hat{\mathbf{y}}, \mathbf{y} \in \{0,1\}^L\):

- \(\mathrm{TP} = \sum_t \mathbb{1}[\hat{y}_t = 1 \land y_t = 1]\), \(\mathrm{PP} = \sum_t \mathbb{1}[\hat{y}_t = 1]\), \(\mathrm{GP} = \sum_t \mathbb{1}[y_t = 1]\).
- Precision \(P = \mathrm{TP}/\mathrm{PP}\) (nếu \(\mathrm{PP}>0\)), Recall \(R = \mathrm{TP}/\mathrm{GP}\) (nếu \(\mathrm{GP}>0\)).
- F-score (harmonic mean):

\[
F_1 = \frac{2PR}{P+R} \quad (\text{khi } P+R>0).
\]

**Temporal overlap** trong code: tạo mask boolean trên timeline từ các đoạn keyframe (pred và GT), rồi:

\[
\text{Overlap} = \frac{|\{t : \text{pred}(t) \land \text{gt}(t)\}|}{|\{t : \text{pred}(t) \lor \text{gt}(t)\}|}
\]

(tương đương IoU frame-level giữa hai tập frame được chọn). Mọi cải tiến làm **điểm số frame khớp GT** hoặc **đoạn keyframe trùng timeline** hơn đều kéo \(P,R,F_1\) và overlap lên.

### 4.3 Loss huấn luyện: BCE + phạt phương sai coverage (Fix 3)

Frame-level logits \(z_{i,t}\) (sau khi huấn luyện tương ứng \(\sigma(z_{i,t})\) là “độ quan trọng”). Phần giám sát chính vẫn là **masked BCE** giữa \(\sigma(z)\) và nhãn frame (chuẩn binary cross-entropy trên từng vị trí hợp lệ).

Bổ sung **diversity / coverage loss**: với mỗi video trong batch, chia \(L\) frame thành \(K\) segment (trong code là `n_segments`). Gọi segment \(j\) có đoạn chỉ số \(S_j\), đặt

\[
m_j = \max_{t \in S_j} \sigma(z_{i,t}).
\]

Loss phụ (trung bình trên batch, bỏ qua video quá ngắn) dùng **phương sai mẫu** của \(\{m_1,\ldots,m_K\}\):

\[
\mathcal{L}_{\text{div}} = \lambda \cdot \mathrm{Var}(m_1,\ldots,m_K), \quad \lambda = \texttt{diversity\_weight}.
\]

**Ý nghĩa:** Nếu model chỉ cho điểm cao ở một vài segment (phương sai lớn), \(\mathcal{L}_{\text{div}}\) lớn → gradient đẩy phân bố \(\max\) trên các segment **đồng đều hơn**, giảm kiểu “chỉ highlight một đoạn” — thường **cải thiện recall** trên GT rải nhiều đoạn và làm binary mask sau top-\(k\) gần GT hơn.

Tổng loss:

\[
\mathcal{L} = \mathcal{L}_{\text{BCE}} + \mathcal{L}_{\text{div}}.
\]

Đây là dạng **đa mục tiêu tổng có trọng số**; \(\lambda\) điều chỉnh trade-off giữa khớp nhãn từng frame và ép phủ timeline.

### 4.4 Kiến trúc: fuse context attention vào scorer (Fix 1)

Gọi \(\mathbf{h}_t \in \mathbb{R}^{d}\) là output BiLSTM tại \(t\), attention cung cấp vector tổng hợp toàn chuỗi \(\mathbf{c} \in \mathbb{R}^{d}\) (weighted sum của các \(\mathbf{h}_t\)). Bản **optimize** dùng đầu vào scorer:

\[
\mathbf{s}_t = f_\theta\big([\mathbf{h}_t \,;\, \mathbf{c}]\big) \in \mathbb{R},
\]

trong khi bản cũ (không fuse) tương đương \(\mathbf{s}_t = f_\theta(\mathbf{h}_t)\) (attention chỉ phục vụ trọng số, không đưa \(\mathbf{c}\) vào \(f_\theta\)).

**Lý luận:** Điểm “quan trọng tương đối toàn video” là hàm của **ngữ cảnh global** lẫn **trạng thái local**; concat \([\mathbf{h}_t;\mathbf{c}]\) cho phép \(f_\theta\) học, ví dụ, “frame này nổi bật so với phần còn lại của clip” — một biến thể **late fusion** cố định, không tăng tham số LSTM nhưng tăng biểu diễn vào lớp cuối. Điều này **trực tiếp** cải thiện chất lượng \(\hat{\mathbf{y}}\) sau top-\(k\), từ đó tăng \(P,R,F_1\).

### 4.5 Inference: hybrid keyshot 70% quality / 30% diversity (Fix 2)

Sau khi có score mượt theo shot, gọi tổng số keyframe cần chọn là \(k_{\text{tot}}\). Chiến lược **hybrid** tách:

- \(k_q \approx \lfloor \alpha k_{\text{tot}}\rceil\) keyframe từ **top score** (quality),
- \(k_d = k_{\text{tot}} - k_q\) từ các shot còn lại theo **chia vùng diversity** (coverage),

với \(\alpha \approx 0.7\) (`quality_ratio`).

So với **chỉ diversity cứng** (mọi segment phải có đại diện): tránh phải chọn frame điểm thấp ở segment “vô nghĩa” → **precision** tốt hơn. So với **chỉ greedy top-\(k\)**: vẫn có vài keyframe “lấp lỗ” timeline → **recall / overlap** tốt hơn. Đây là **trade-off có điều khiển** giữa hai cực; F-score (harmonic mean) thường hưởng lợi khi cả hai nhánh không bị một chiến lược đơn thuần làm lệch quá mạnh.

### 4.6 Scheduler & weight decay (Fix 4)

- **`ReduceLROnPlateau`**: giảm learning rate khi `val_loss` không cải thiện sau `patience` epoch — tương đương bước **fine-tuning** trong vùng lân cận cực trị, thường giúp hội tụ ổn định hơn so với `StepLR` cố định chu kỳ (không phụ thuộc mức loss).
- **`weight_decay: 1e-4`** (so với `1e-5`): regularization \(L_2\) mạnh hơn trên trọng số, giảm overfit trên tập train → **generalization** trên test tốt hơn nếu trước đó model quá khớp.

### 4.7 Tóm tắt mối liên hệ “thay đổi → metric”

| Thành phần | Tác động lên metric (định tính) |
|------------|----------------------------------|
| Fuse \(\mathbf{c}\) vào scorer | Score frame khớp GT tốt hơn → \(F_1\) ↑ |
| \(\mathcal{L}_{\text{div}}\) (variance của max theo segment) | Phủ timeline, ít bỏ sót đoạn → Recall ↑, overlap ↑ |
| Hybrid keyshot | Cân quality/coverage → cả Precision và Recall đỡ cực đoan |
| Plateau + weight decay | Val/test ổn định hơn, ít overfit → metric test ↑ |
| Load đúng `fuse_attention_context` khi eval | So sánh đúng kiến trúc đã train — bắt buộc để kết luận hợp lệ |

Nếu cần **chứng minh định lượng** tách từng nhân tố (ablation), nên giữ nguyên data split và chỉ bật/tắt từng Fix trong config — một thí nghiệm đơn, một bảng số; phần toán học trên chỉ mô tả **cơ chế** và **hướng** cải thiện, không thay cho ablation trên dữ liệu của bạn.
