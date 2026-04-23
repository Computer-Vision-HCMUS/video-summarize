# Multimodal Video Summarization via Supervised Learning

**Project name:** `video-summarize`  
**Problem:** Importance-based **video summarization** — predict which frames or segments to keep so that a shorter **dynamic summary** preserves key content and narrative, with optional **multimodal** (vision + speech) cues.  
**Model:** **ResNet-50** frame encoder · **Whisper** + **Sentence-BERT** speech branch · **two-layer BiLSTM** · **temporal attention** · MLP scorer  
**Datasets:** **SumMe** (user videos, multi-annotator summaries) · **TVSum** (50 videos, crowdsourced importance)  
**Framework:** **PyTorch** (≥ 2.0)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-3670A0?logo=python&logoColor=ffdd54)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-≥2.0-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)

---

## 1. Introduction

### 1.1 Problem and motivation

Long-form video dominates web and mobile traffic; fully watching every clip is costly for users and downstream systems. **Video summarization** aims to produce a compact surrogate—either a set of keyframes (**static** summary) or a shorter video (**dynamic** summary with aligned audio)—that retains salient events and semantics.

This repository targets **supervised, importance-based** summarization: the model learns to match **frame-level importance scores** derived from human annotations. The problem matters for browsing, retrieval, education, and as a precursor to efficient **multimodal analysis** (search, QA) on summarized content.

### 1.2 Approach (overview)

1. **Sampling & encoding:** Frames are sampled at a fixed rate (default **2 fps**). Visual vectors \(\mathbf{v}_t \in \mathbb{R}^{D_v}\) come from a pretrained **ResNet-50** (\(D_v=2048\)). Optionally, audio is transcribed with **Whisper** and encoded with **Sentence-BERT**, yielding \(\mathbf{a}_t \in \mathbb{R}^{D_a}\) (\(D_a=384\)) aligned to the same time grid.
2. **Sequence model:** Concatenated features \(\mathbf{x}_t = [\mathbf{v}_t; \mathbf{a}_t] \in \mathbb{R}^{2432}\) are fed to a **two-layer BiLSTM** with **temporal attention** to produce per-frame logits \(z_t\).
3. **Training:** Scores are trained against continuous labels with **masked BCE with logits**, plus an optional **diversity / coverage** term on sigmoid scores across temporal segments.
4. **Inference:** Logits are smoothed, converted to keyframes/shots under budget \(r\), and exported with **FFmpeg** so audio can be preserved.

A full mathematical write-up, related work, and **VideoIntel** downstream evaluation appear in [`docs/report/report.tex`](docs/report/report.tex) / [`docs/report/report.pdf`](docs/report/report.pdf).

---

## 2. Model Architecture

### 2.1 Multimodal feature fusion

At time step \(t\):

$$
\mathbf{x}_t = \begin{bmatrix} \mathbf{v}_t \\ \mathbf{a}_t \end{bmatrix} \in \mathbb{R}^{D_v + D_a}, \quad D_v = 2048,\; D_a = 384.
$$

Visual-only training uses \(\mathbf{x}_t = \mathbf{v}_t\) only (dimension matches the checkpoint / feature files).

### 2.2 BiLSTM + temporal attention + scorer

Let \(\mathbf{h}_t \in \mathbb{R}^{2h}\) denote the BiLSTM output at \(t\) (forward and backward hidden size \(h=256\) per direction). **Temporal attention** computes a context vector \(\mathbf{c}\) over \(\{\mathbf{h}_1,\ldots,\mathbf{h}_T\}\). The implementation can **fuse** \(\mathbf{c}\) with each \(\mathbf{h}_t\) before a two-layer MLP outputs a scalar logit \(z_t\) (importance score). When attention is ablated, the scorer uses \(\mathbf{h}_t\) alone.

**Forward (schematic):**

$$
\{\mathbf{h}_t\} = \mathrm{BiLSTM}(\{\mathbf{x}_t\}), \quad
(\mathbf{c}, \boldsymbol{\alpha}) = \mathrm{Attn}(\{\mathbf{h}_t\}), \quad
z_t = f_{\mathrm{mlp}}([\mathbf{h}_t; \mathbf{c}]).
$$

### 2.3 Diagram (placeholder)

**Figure 1 — High-level pipeline (replace with your own figure).**

```text
┌─────────────┐     ┌──────────────┐     ┌─────────────────────────┐
│ Video frames│────►│ ResNet-50    │────►│ v_t ∈ R^{2048}          │
└─────────────┘     └──────────────┘     └───────────┬───────────────┘
                                                     │ concat
┌─────────────┐     ┌──────────────┐     ┌───────────▼───────────────┐
│ Audio track │────►│ Whisper+SBERT│────►│ a_t ∈ R^{384}           │
└─────────────┘     └──────────────┘     └───────────┬───────────────┘
                                                     │
                        ┌────────────────────────────▼────────────────┐
                        │ BiLSTM (2×256) → Temporal Attention → MLP   │
                        └────────────────────────────┬────────────────┘
                                                     │ z_t
                        ┌────────────────────────────▼────────────────┐
                        │ Smoothing · keyshots · FFmpeg → V_sum       │
                        └─────────────────────────────────────────────┘
```

For publication-quality figures, export TikZ or PDF from `docs/report/report.tex` and link them here.

---

## 3. Dataset

| Dataset | Content | Labels | Typical use in this repo |
|--------|---------|--------|---------------------------|
| **SumMe** [1] | 25 user-style videos; events, sports, holidays | Multiple human summaries / frame-level scores (after preprocessing) | Train / eval split with project JSON labels |
| **TVSum** [2] | 50 videos (e.g. news, how-to) | Crowdsourced frame/shot importance | Often identified by `video_id` prefix `video_*` in prepared files |

**Preprocessing (conceptual):**

1. Obtain raw videos or ECCV16-style **feature `.h5`** files; scripts under `scripts/` prepare **`data/labels/*.json`** (per-frame scores in \([0,1]\)) and **`data/features/*.npy`**.
2. Optional: **`{id}_audio.npy`** from Whisper + Sentence-BERT, time-aligned to the visual grid.
3. **Metadata:** `data/features/_meta.json` records `feature_dim` for training.

**Splits:** Default **80% / 10% / 10%** train / val / test with fixed `seed` in `configs/config.yaml` (adjust for official folds if you compare to specific papers).

---

## 4. Training

### 4.1 Hyperparameters (defaults, `configs/config.yaml`)

| Group | Parameter | Value |
|-------|-----------|--------|
| Data | `max_seq_len` | 960 (~8 min @ 2 fps) |
| Data | train / val / test | 0.8 / 0.1 / 0.1 |
| Model | BiLSTM `hidden_size` | 256 (per direction) |
| Model | `num_layers` | 2 |
| Model | `dropout` | 0.3 |
| Model | `bidirectional` | true |
| Model | `use_attention` | true |
| Training | `batch_size` | 4 |
| Training | `epochs` | 50 |
| Training | `learning_rate` | \(10^{-3}\) |
| Training | `weight_decay` | \(10^{-4}\) |
| Training | `gradient_clip` | 1.0 |
| Training | `early_stopping_patience` | 10 |
| Training | `diversity_weight` | 0.3 |
| Training | `n_segments` | 5 |
| Optimizer | Scheduler | `ReduceLROnPlateau` (factor 0.5, patience 5, `min_lr` \(10^{-5}\)) |
| Inference | `summary_ratio` \(r\) | **0.35** |

### 4.2 Loss function

**Primary:** masked **binary cross-entropy with logits** between predicted scores and continuous ground-truth scores on valid (non-padded) positions:

$$
\mathcal{L}_{\mathrm{BCE}} = \frac{1}{\sum_i T_i} \sum_{i,t \leq T_i} \mathrm{BCEWithLogits}(z_{i,t}, y_{i,t}).
$$

**Auxiliary (diversity / coverage):** a penalty encourages spread of high scores across \(N\) temporal segments (implementation in `src/training/loop.py`):

$$
\mathcal{L} = \mathcal{L}_{\mathrm{BCE}} + \lambda_{\mathrm{div}} \mathcal{L}_{\mathrm{coverage}}, \quad \lambda_{\mathrm{div}} = 0.3.
$$

### 4.3 Training strategy

- **Optimizer:** AdamW.
- **Checkpointing:** best validation loss → `checkpoints/best.pt`; periodic `epoch_*.pt` every 5 epochs.
- **Reproducibility:** `seed: 42` via `set_seed()` for Python / NumPy / PyTorch.
- **Multimodal:** use `scripts/train_multimodal.py` so `input_dim = D_v + D_a\).

---

## 5. Results

### 5.1 Metrics

Evaluation (`scripts/run_eval.py`) reports **mean precision, recall, F-score**, and **temporal overlap** on the held-out split. Frame-level predictions and binarized ground truth use a **top-\(k\)** rule with the same **`summary_ratio`** \(r\) as inference, so reported precision and recall can be numerically close; interpret F-score **per dataset** (SumMe vs TVSum) separately.

### 5.2 Empirical ablation table (one replication)

The following numbers are produced by the project’s evaluator and **one training run per variant** (see [`docs/report/abalation.md`](docs/report/abalation.md)). They are **not** directly comparable to published tables unless splits, \(r\), smoothing, and shot logic match those papers.

| Model / variant | SumMe F1 (%) | TVSum F1 (%) | Precision (%) | Recall (%) | Temporal overlap (%) |
|-----------------|-------------:|-------------:|--------------:|-----------:|---------------------:|
| Multimodal BiLSTM + temporal attention (full) | 26.9 | 60.1 | 47.7 | 47.7 | 23.5 |
| Visual-only (no audio) | 31.4 | 59.7 | 49.1 | 49.1 | 33.3 |
| BiLSTM, no temporal attention | 34.9 | 63.7 | 52.9 | 52.9 | 27.4 |
| UniLSTM (`bidirectional=false`) | 35.4 | 63.1 | 52.7 | 52.7 | 37.3 |

**Discussion (brief):** On this run, simpler variants score higher on **SumMe** F1 than the full multimodal model—consistent with optimization difficulty, audio noise, or protocol sensitivity. Report **multiple seeds** and confidence intervals for stronger claims. **TVSum** F1 is higher than SumMe across rows, reflecting dataset and label differences.

### 5.3 Comparison with literature (protocol caution)

Published F-scores on SumMe/TVSum vary widely with **split type**, **\(r\)**, and post-processing. For a curated literature table, see **`docs/report/report.md`** or **Section 8** of `docs/report/report.tex`. **Do not** claim SOTA without matching protocols.

---

## 6. Ablation Study

| Component removed or changed | Research question | Observed trend (this repo, Table §5.2) |
|-----------------------------|-------------------|----------------------------------------|
| **Audio / SBERT branch** (visual-only) | Does speech add complementary saliency? | SumMe F1 ↑ vs full in table; TVSum ≈ — suggests **nuanced** role of audio under current training |
| **Temporal attention** | Is global context fusion necessary? | Higher SumMe/TVSum F1 **without** attention in this run — investigate **fusion-in-scorer** vs simpler inductive bias |
| **Bidirectionality** (UniLSTM) | How much does future context help? | Strong SumMe/TVSum F1 — worth comparing with **matched training budget** for fair conclusion |

**Tooling:** `train-abalation.py` trains variants; `abalation.py --mode variants` builds Markdown tables.

---

## 7. Installation

```bash
git clone <repository-url>
cd video-summarize
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install openai-whisper sentence-transformers   # multimodal
```

Install **FFmpeg** and ensure it is on `PATH` (audio + summary export).

---

## 8. Usage

**Training (visual-only):**

```bash
python -m scripts.train --config configs/config.yaml
```

**Training (multimodal):**

```bash
python -m scripts.extract_features --config configs/config.yaml --audio --whisper-model base
python -m scripts.train_multimodal --config configs/config.yaml
```

**Evaluation:**

```bash
python -m scripts.run_eval --config configs/config.yaml --checkpoint checkpoints/best.pt
```

**Inference / demo:**

```bash
python -m streamlit run app.py
```

**Ablations:**

```bash
python train-abalation.py --config configs/config.yaml
python abalation.py --mode variants --config configs/config.yaml \
  --full checkpoints/best.pt \
  --visual-only checkpoints/ablation_visual_only.pt \
  --no-attention checkpoints/ablation_no_attention.pt \
  --unilstm checkpoints/ablation_unilstm.pt \
  --output docs/report/abalation.md
```

---

## 9. Project Structure

| Path | Role |
|------|------|
| `configs/config.yaml` | Hyperparameters and paths |
| `data/{raw,features,labels}/` | Videos, `.npy` features, JSON labels |
| `src/{data,features,models,training,evaluation,inference}/` | Core library |
| `scripts/` | CLI: download, extract, train, eval, inference |
| `app.py` | Streamlit UI |
| `train-abalation.py`, `abalation.py` | Ablation train + report |
| `docs/report/` | IEEE-style manuscript (`report.tex`, PDF, `abalation.md`) |
| `checkpoints/` | Saved `.pt` (typically gitignored) |

---

## 10. Future Work

- **Stronger multimodal fusion:** cross-attention or transformer encoders over \(\{\mathbf{x}_t\}\); alignment losses (cf. multimodal summarization literature).
- **Evaluation protocol:** GT binarization independent of prediction budget; reporting **mAP** or **rank correlation** with continuous scores; **multi-seed** tables.
- **Data:** official SumMe/TVSum splits for strict comparison; transfer to egocentric or long-form video.
- **Efficiency:** distill BiLSTM scorer; compress ResNet features or use lighter backbones.
- **Downstream:** extend **VideoIntel**-style QA/search evaluation with standardized benchmarks.

---

## 11. References

[1] M. Gygli, H. Grabner, H. Riemenschneider, and L. Van Gool, “Creating summaries from user videos,” in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2014, pp. 505–520. [DOI: 10.1007/978-3-319-10584-0_33](https://doi.org/10.1007/978-3-319-10584-0_33)

[2] Y. Song, J. Vallmitjana, A. Stent, and A. Jaimes, “TVSum: Summarizing web videos using titles,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2015, pp. 5179–5187. [DOI: 10.1109/CVPR.2015.7299154](https://doi.org/10.1109/CVPR.2015.7299154)

[3] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in *Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR)*, 2016. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385)

[4] K. Zhang, W.-L. Chao, F. Sha, and K. Grauman, “Video summarization with long short-term memory,” in *Proc. Eur. Conf. Comput. Vis. (ECCV)*, 2016. [DOI: 10.1007/978-3-319-46493-0_47](https://doi.org/10.1007/978-3-319-46493-0_47)

[5] J. Fajtl, H. S. Sohák, A. Argyriou, and D. Monekosso, “Summarizing videos with attention,” in *Proc. Asian Conf. Comput. Vis. (ACCV)*, 2018. [arXiv:1812.01969](https://arxiv.org/abs/1812.01969)

[6] A. Radford *et al.*, “Robust speech recognition via large-scale weak supervision,” in *Proc. Int. Conf. Mach. Learn. (ICML)*, 2023. (Whisper.) [arXiv:2212.04356](https://arxiv.org/abs/2212.04356)

[7] N. Reimers and I. Gurevych, “Sentence-BERT: Sentence embeddings using Siamese BERT-networks,” in *Proc. Conf. Empirical Methods Nat. Lang. Process. (EMNLP)*, 2019. [arXiv:1908.10084](https://arxiv.org/abs/1908.10084)

---

**Authors (manuscript):** Tran Hai Duc, Nguyen Cong Chien — University of Science, VNU-HCM (see `docs/report/report.tex`).  
**License:** specify in a root `LICENSE` file if redistributing; `pyproject.toml` lists package metadata only.
