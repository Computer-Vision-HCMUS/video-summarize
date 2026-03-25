# So Sánh Kết Quả Mô Hình

## Mô hình của tôi

| Thuộc tính        | Chi tiết                                                     |
| ----------------- | ------------------------------------------------------------ |
| Checkpoint        | `checkpoints/best.pt`                                        |
| Kiến trúc         | Multimodal BiLSTM + Temporal Attention                       |
| Backbone ảnh      | ResNet-50 (dim = 2048)                                       |
| Feature âm thanh  | Whisper + SentenceBERT (dim = 384)                           |
| Input dim         | 2432 (= 2048 + 384)                                          |
| Frame sampling    | 2 fps, `max_seq_len` = 960                                   |
| Smoothing         | Gaussian (sigma theo config)                                 |
| Summary ratio `r` | 0.35                                                         |
| Shot segmentation | `keyshot.py` (shot_len / min_shot_len / min_gap / diversity) |
| Segment expansion | ±1.5s                                                        |

---

## 1. Kết quả của tôi

| Mô hình / Biến thể                                      | SumMe F-score (%) | TVSum F-score (%) | Precision (%) | Recall (%) | Temporal Overlap |
| ------------------------------------------------------- | ----------------: | ----------------: | ------------: | ---------: | ---------------: |
| **Multimodal BiLSTM + Temporal Attention** (full model) |                   |                   |               |            |                  |
| Ablation: Visual-only BiLSTM (bỏ audio)                 |                   |                   |               |            |                  |
| Ablation: BiLSTM không có Temporal Attention            |                   |                   |               |            |                  |
| Ablation: UniLSTM (bidirectional = False)               |                   |                   |               |            |                  |

> **Điền vào sau khi chạy inference.** Tất cả các biến thể cần chạy cùng protocol (cùng split, cùng `r`, cùng keyshot policy).

---

## 2. So sánh với các phương pháp từ literature

> Số liệu dưới đây là **F-score (%)** được báo cáo trong từng paper. Lưu ý: các paper có thể dùng protocol khác nhau — hãy đọc cột _Ghi chú_ trước khi so sánh trực tiếp.

|   # | Phương pháp                    |  Năm | Loại         |  SumMe (%) |  TVSum (%) | Ghi chú                                     |
| --: | ------------------------------ | ---: | ------------ | ---------: | ---------: | ------------------------------------------- |
|   1 | vsLSTM (Zhang et al.)          | 2016 | Supervised   |       37.6 |       54.2 | Random split, bảng tổng hợp PGL-SUM         |
|   2 | dppLSTM (Zhang et al.)         | 2016 | Supervised   |       38.6 |       54.7 | Random split, bảng tổng hợp PGL-SUM         |
|   3 | SUM-GAN (Mahasseni et al.)     | 2017 | Unsupervised |       38.7 |       50.8 | Table 1, σ=0.3                              |
|   4 | Cycle-SUM (Zhang et al.)       | 2019 | Unsupervised |       41.9 |       57.6 | Avg 5 random splits                         |
|   5 | VASNet (Fajtl et al.)          | 2018 | Supervised   |       49.6 |       61.4 | Standard splits                             |
|   6 | DSNet anchor-free (Zhu et al.) | 2020 | Supervised   |       51.2 |       61.9 | Theo repo/paper DSNet                       |
|   7 | MSVA (Ghauri et al.)           | 2021 | Supervised   |       53.4 |       61.5 | Non-overlapping splits                      |
|   8 | PGL-SUM (Apostolidis et al.)   | 2021 | Supervised   |       57.1 |       62.7 | 5 random splits                             |
|   9 | SUM-GDA                        | 2020 | —            |       56.3 |       60.7 | Xem Table trong paper                       |
|  10 | MHSCNet                        | 2022 | Supervised   |          — |          — | Nhiều setting (Standard/Aug/Transfer)       |
|   — | **Mô hình của tôi**            |    — | Supervised   | **(điền)** | **(điền)** | r=0.35, Gaussian smoothing, ±1.5s expansion |

---

## 3. Lưu ý khi so sánh

Hai yếu tố ảnh hưởng lớn nhất đến F-score khi so sánh giữa các paper:

- **Dataset split / protocol**: random splits hay official folds, bao nhiêu folds, seed.
- **Summary ratio `r`** và **cách chọn keyshot**: smoothing, shot grouping, segment expansion.

Nếu protocol của tôi khác với paper gốc, chỉ nên ghi nhận con số tham khảo, không kết luận trực tiếp hơn/kém.

---

## 4. Nguồn tham khảo

| Paper            | Link                                                                                    |
| ---------------- | --------------------------------------------------------------------------------------- |
| vsLSTM / dppLSTM | Zhang et al., ECCV 2016 + bảng PGL-SUM                                                  |
| SUM-GAN          | https://web.engr.oregonstate.edu/~sinisa/research/publications/cvpr17_summarization.pdf |
| Cycle-SUM        | https://ojs.aaai.org/index.php/AAAI/article/view/4948/4821                              |
| VASNet           | https://github.com/azhar0100/VASNet                                                     |
| DSNet            | https://github.com/li-plus/DSNet                                                        |
| MSVA             | https://arxiv.org/pdf/2104.11530.pdf                                                    |
| PGL-SUM          | https://www.iti.gr/~bmezaris/publications/ism2021a_preprint.pdf                         |
| MHSCNet          | https://arxiv.org/pdf/2204.08352                                                        |
