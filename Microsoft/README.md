# GoldenASR — Golden Transcription Selection

**Hackathon: Renan Partners — Arabic ASR Quality**

Given N audio samples, each with 5 candidate transcriptions, automatically select the highest-quality ("golden") transcription per audio.

---

## What We Changed

### v1: Whisper-based Pipeline (`kaggle_notebook.py`)

- **Scorer**: OpenAI Whisper `large-v3` (1.55B parameters)
- **Evaluator**: Whisper `medium` (independent, breaks circularity)
- **Approach**: Transcribe audio with Whisper, pick the candidate closest to the transcription (Levenshtein similarity)
- **Issue**: Whisper large-v3 is heavy (~6GB VRAM), prone to OOM on Kaggle T4 GPUs, and slow

### v2: Moonshine-based Pipeline (`new_notebook.py`)

- **Scorer**: Moonshine `tiny-ar` (27M parameters) — purpose-built for Arabic
- **Evaluator**: Whisper `medium` (unchanged, different model family = no circularity)
- **Key improvements**:
  - **55× fewer parameters** (27M vs 1.55B) — no OOM risk
  - **Multilingual model map** — auto-selects the right Moonshine model per language (`ar`, `zh`, `ja`, `ko`, `uk`, `vi`)
  - **Hallucination guard** — `token_limit_factor` caps output length per language
  - **Faster loading** — 108MB download vs ~3GB for Whisper large-v3
  - Uses HuggingFace Transformers API (`MoonshineForConditionalGeneration`)

---

## Results Comparison

### Independent Evaluation (Whisper Medium as Proxy Ground Truth)

#### Audio-Based Scorers (the ones that changed)

| Metric | Whisper large-v3 (v1) | Moonshine tiny-ar (v2) | Change |
|--------|:---------------------:|:----------------------:|:------:|
| **Accuracy** | **42.0%** | 40.0% | -2% |
| **WER** | 0.8761 | 0.8762 | ~same |
| **CER** | 0.6572 | **0.6393** | **-2.7%** ✅ |
| **Gap to Oracle** | **0.0272** | 0.0396 | +0.012 |

#### Confidence-Gated Hybrid (Moonshine/Whisper when confident, MBR when uncertain)

| Metric | v1 (Whisper) | v2 (Moonshine) | Change |
|--------|:------------:|:--------------:|:------:|
| **Accuracy** | 40.0% | 40.0% | same |
| **WER** | 0.8777 | **0.8740** | **-0.4%** ✅ |
| **CER** | 0.6576 | **0.6389** | **-2.8%** ✅ |
| **Gap to Oracle** | **0.0288** | 0.0374 | +0.009 |

#### All Scorers (v2 — Moonshine)

| Scorer | Acc | WER | CER | Oracle | Gap |
|--------|-----|-----|-----|--------|-----|
| MBR (vanilla) | 17.0% | 0.8963 | 0.7115 | 0.8366 | 0.0597 |
| MBR + Outlier | 18.0% | 0.8923 | 0.7074 | 0.8366 | 0.0557 |
| Text Heuristics | 25.0% | 0.8885 | 0.7104 | 0.8366 | 0.0520 |
| **Moonshine Pseudo-Ref** | **40.0%** | **0.8762** | **0.6393** | 0.8366 | 0.0396 |
| **Confidence-Gated** | **40.0%** | **0.8740** | **0.6389** | 0.8366 | 0.0374 |
| Full Fusion | 31.0% | 0.8810 | 0.6821 | 0.8366 | 0.0444 |

### Key Takeaway

> Moonshine tiny-ar (27M params) achieves **nearly identical accuracy** and **better CER** compared to Whisper large-v3 (1.55B params) — using **55× fewer parameters**, with **no OOM issues** and **faster inference**.

---

## Scoring Strategies

| Scorer | What it does | GPU? | Role |
|--------|-------------|------|------|
| MBR (vanilla) | Pairwise CER consensus across 5 candidates | No | Baseline |
| MBR + Outlier | MBR + drama script detection | No | Baseline |
| Text Heuristics | MBR + length normality + word count | No | Baseline |
| **Moonshine Pseudo-Ref** | Similarity to Moonshine transcription | Yes | **Primary** |
| **Confidence-Gated** | Moonshine when confident, MBR when uncertain | Yes | **Backup** |
| Full Fusion | Weighted combination of all signals | Yes | Comparison |

---

## How to Run on Kaggle

1. Upload `new_notebook.py` to Kaggle
2. **Settings**:
   - Accelerator: **GPU T4 x2**
   - Internet: **ON** (downloads Moonshine model from HuggingFace)
3. Paste each `# ========== BLOCK N ==========` section into a code cell
4. Run all cells sequentially
5. Outputs are saved to `/kaggle/working/output/`

---

## Files

| File | Description |
|------|-------------|
| `new_notebook.py` | **Current** — Moonshine-based pipeline (v2) |
| `kaggle_notebook.py` | Previous Whisper-based pipeline (v1) |
| `new_result.txt` | Results from Moonshine run |
| `result.txt` | Results from Whisper run |
| `TECHNICAL_REPORT.md` | Technical details of the scoring architecture |
