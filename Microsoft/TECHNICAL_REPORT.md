# GoldenASR — Technical Report

## 1. Problem Statement

Given N audio samples, each with 5 candidate transcriptions from different annotators/ASR systems, automatically select the highest-quality ("golden") transcription per audio and compute WER of all candidates against it.

## 2. System Architecture

```
Audio File
    │
    ▼
┌──────────────────────────────────────────────────┐
│  Shared Whisper Encoder (large-v3, runs ONCE)    │
└──────────────────┬───────────────────────────────┘
                   │
      ┌────────────┼─────────────────┐
      ▼            ▼                 ▼
 [CTC Forced   [Speculative     [Consensus
  Alignment]    Rescoring:        MBR: pairwise
  wav2vec2      small draft →     CER across 5
  XLSR-53       large verify]     candidates]
      └────────────┼─────────────────┘
                   ▼
         ┌──────────────────┐
         │   LM Fluency     │
         │   AraGPT2        │
         │   (EAGLE-style   │
         │    KV cache)     │
         └────────┬─────────┘
                  ▼
        Weighted Score Fusion
        → argmax = Golden
        → WER vs Golden
        → Output CSV
```

## 3. Scoring Components

### 3.1 CTC Forced Alignment (weight: 0.35)
- **Model:** facebook/wav2vec2-large-xlsr-53 (multilingual CTC)
- **Method:** CTC forward algorithm (dynamic programming) aligns each candidate character-by-character to the audio frames
- **Score:** Average log-probability along the best alignment path, normalized by transcription length
- **Fallback:** Greedy frame-matching if DP fails

### 3.2 Whisper Speculative Rescoring (weight: 0.35)
- **Models:** whisper-large-v3 (verifier) + whisper-small (draft)
- **Method:**
  1. Shared encoder encodes audio ONCE (saves 4× encoder passes)
  2. Draft model (small) scores all 5 candidates via teacher-forcing in a single batch
  3. Only top-K candidates (default K=2) proceed to the large verifier
  4. Rejected candidates receive their draft score with a penalty offset
- **Score:** Average per-token log-probability under teacher-forced decoding

### 3.3 Consensus MBR (weight: 0.15)
- **Model:** None (text-only, zero compute)
- **Method:** Pairwise character error rate across all 5 candidates. The transcription closest to all others (lowest average CER) scores highest.
- **Score:** `1 - avg_CER` ∈ [0, 1]
- **Rationale:** Errors made by a single annotator get outvoted by the majority

### 3.4 LM Fluency — EAGLE-style (weight: 0.15)
- **Model:** aubmindlab/aragpt2-base
- **Method:**
  1. Tokenize all 5 candidates
  2. Find shared token prefix across candidates
  3. One forward pass on shared prefix → cache KV states
  4. Score each candidate's unique suffix using cached prefix KV
  5. ~3-4× speedup vs independent scoring
- **Score:** Average negative log-likelihood (lower = more fluent; fusion inverts)

## 4. Score Fusion

All signals are min-max normalized to [0, 1]. LM scores are inverted (lower NLL → higher normalized).

```
S_final(h) = w_acoustic · Ŝ_acoustic + w_recscore · Ŝ_recscore + w_mbr · Ŝ_mbr + w_lm · (1 - Ŝ_lm)
```

Default weights: `[0.35, 0.35, 0.15, 0.15]`

Golden = argmax over all candidates.

## 5. Optimizations

| Optimization | Impact |
|---|---|
| Shared Whisper encoder (encode once) | 5× → 1× encoder passes |
| Speculative rescoring (small→large) | ~60% fewer large model calls |
| Cascade filtering (top-K=2) | Only top-2 ambiguous candidates go to expensive stages |
| EAGLE prefix-tree KV cache reuse | ~3× fewer LM forward passes |
| Parallel audio download (ThreadPool) | Near-zero download bottleneck |
| Lazy module imports | CPU-only mode runs without GPU deps loaded |

## 6. Input/Output Format

**Input CSV:** `audio_id, language, audio (URL), option_1, ..., option_5`

**Output CSV:** Input columns + `golden_ref, golden_option, golden_score, wer_option1, ..., wer_option5`

## 7. Running

```bash
# Full GPU pipeline
python main.py --input data/input.csv --output output/results.csv --language ar

# CPU-only (consensus MBR + WER only)
python test_smoke.py

# Lightweight pipeline (no audio/GPU needed)
python -c "
from golden_asr.config import Config
from golden_asr.pipeline_lite import LightweightPipeline
pipe = LightweightPipeline(Config())
pipe.run('data/sample_input.csv')
"

# Run tests
python -m pytest tests/ -v
```

## 8. Results

<!-- Fill in after running on the full dataset -->

| Metric | Value |
|---|---|
| Total audio samples | — |
| Processing time | — |
| Avg time per sample | — |
| Golden selection distribution | — |
| Avg WER option_1 vs golden | — |
| Avg WER option_2 vs golden | — |
| Avg WER option_3 vs golden | — |
| Avg WER option_4 vs golden | — |
| Avg WER option_5 vs golden | — |

## 9. Limitations & Future Work

- CTC alignment uses character-level vocab; performance may vary for languages with large character sets
- Fusion weights are hand-tuned; could be learned on a held-out validation set
- No batched processing across audio samples yet (sequential per-sample)
- EAGLE KV reuse benefit depends on how much prefix candidates share
