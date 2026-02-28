# GoldenASR Benchmark

**One command to test your scorer. One command to compare everyone's results.**

## Quick Start (30 seconds)

```bash
# 1. Copy the template
cp scorers/template.py scorers/yourname.py

# 2. Edit scorers/yourname.py — implement the score() method

# 3. Run benchmark
python run_benchmark.py scorers.yourname.YourClassName --data data/input.csv

# 4. Compare all results
python compare_results.py
```

## How It Works

You write a scorer class. The benchmark:
1. Loads the CSV dataset
2. Downloads audio (cached after first run)
3. Calls your `score()` for each row
4. Picks the option with the highest score as golden
5. Computes WER/CER of all other options vs golden
6. Saves results CSV + metrics JSON

## Write Your Scorer

```python
# scorers/yourname.py
from typing import List, Optional
from benchmark.base_scorer import BaseScorer

class YourScorer(BaseScorer):

    @property
    def name(self) -> str:
        return "yourname_v1"          # unique name, used in output filenames

    def setup(self):
        pass                          # load models here (called once)

    def score(self, options: List[str], audio_path: Optional[str] = None, language: str = "ar") -> List[float]:
        # Your logic here. Return 5 floats — highest = best.
        return [0.5, 0.8, 0.3, 0.1, 0.2]
```

### What you receive

| Arg | Type | Description |
|-----|------|-------------|
| `options` | `List[str]` | The 5 transcription candidates |
| `audio_path` | `str \| None` | Local path to the `.wav` file (16kHz) |
| `language` | `str` | ISO code: `"ar"`, `"en"`, etc. |

### What you return

`List[float]` — 5 scores, one per option. **Highest score wins.**

## Run Benchmark

```bash
# Full run
python run_benchmark.py scorers.yourname.YourScorer --data data/input.csv

# Quick test (first 10 samples only)
python run_benchmark.py scorers.yourname.YourScorer --data data/input.csv --limit 10

# Text-only scorer (skip audio download)
python run_benchmark.py scorers.yourname.YourScorer --data data/input.csv --no-audio

# Custom output dir
python run_benchmark.py scorers.yourname.YourScorer -o my_results/
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--data`, `-d` | `data/input.csv` | Input CSV path |
| `--audio-dir` | `data/audio` | Audio cache dir |
| `--output`, `-o` | `benchmark_results` | Output dir |
| `--limit`, `-n` | all | First N samples only |
| `--no-audio` | false | Skip audio download |

## Compare Results

After multiple team members have run their scorers:

```bash
python compare_results.py

# Sort by a specific metric
python compare_results.py --sort-by avg_wer_others_vs_golden
```

Output:
```
  SCORER COMPARISON
---------------------------------------------------
  Metric                         consensus_baseline    whisper_v3
---------------------------------------------------
  Samples                                        50                 50
  Total time (s)                                0.1               45.2
  Avg time/sample (s)                        0.0020             0.9040
---------------------------------------------------
  MBR agreement                              100.0%              78.0%
  Avg WER others vs golden                   0.1832             0.1523
  Avg CER others vs golden                   0.0921             0.0742
  Confidence margin                          0.0512             0.2341
  Cross-lang std(WER)                        0.0000             0.0132
---------------------------------------------------
  Per-Language WER
    ar                                       0.1832             0.1523
    en                                          --              0.1401
```

## Pre-built Scorers

### Text-only (no GPU needed)

```bash
# MBR consensus baseline (instant, zero-model)
python run_benchmark.py scorers.consensus_baseline.ConsensusBaseline --data data/input.csv --no-audio

# Text heuristics (MBR + length + diversity + word count)
python run_benchmark.py scorers.text_heuristics.TextHeuristicsScorer --data data/input.csv --no-audio

# Random (sanity check — you must beat this)
python run_benchmark.py scorers.random_baseline.RandomScorer --data data/input.csv --no-audio
```

### Audio + Text (GPU required)

```bash
# Whisper teacher-forced rescoring (with speculative draft→verify)
python run_benchmark.py scorers.whisper_rescore.WhisperRescorer --data data/input.csv

# Whisper generate-and-compare reference
python run_benchmark.py scorers.whisper_reference.WhisperRefScorer --data data/input.csv
python run_benchmark.py scorers.whisper_reference.WhisperRefSmall --data data/input.csv

# CTC forced alignment (wav2vec2-xlsr-53)
python run_benchmark.py scorers.ctc_align.CTCAlignScorer --data data/input.csv

# LM perplexity (AraGPT2, text-only but needs GPU for speed)
python run_benchmark.py scorers.lm_perplexity.LMPerplexityScorer --data data/input.csv --no-audio
```

### Fusion Ensembles (combined signals)

```bash
# Full pipeline: Whisper + CTC + MBR + LM
python run_benchmark.py scorers.fusion_ensemble.FullPipelineScorer --data data/input.csv

# Text-only fusion: MBR + LM
python run_benchmark.py scorers.fusion_ensemble.TextOnlyFusionScorer --data data/input.csv --no-audio

# Acoustic-only: Whisper + CTC
python run_benchmark.py scorers.fusion_ensemble.AcousticOnlyScorer --data data/input.csv
```

## Output Files

Each run produces 3 files in `benchmark_results/`:

| File | Purpose |
|------|---------|
| `{name}_results.csv` | Submission-format CSV (golden_ref, WER per option) |
| `{name}_metrics.json` | All metrics as JSON (for automated comparison) |
| `{name}_details.json` | Per-sample scores (for debugging) |

## Metrics Explained

### Two Evaluation Modes

**The hackathon dataset has NO ground-truth labels** — just 5 candidate transcriptions per audio. Without knowing which is "correct," we can only measure proxy signals (scorer agreement, pick distinctness). If you find or create a labeled dataset, the benchmark automatically switches to real accuracy evaluation.

#### Mode 1: Proxy Metrics (default — no labels)

These do NOT measure accuracy. They measure internal consistency and can help compare scorers against each other.

| Metric | What it actually measures | Better |
|--------|--------------------------|--------|
| **MBR agreement** | Does this scorer pick the same option as text consensus? | Context-dependent |
| **Pick distinctness (WER)** | How different are the 4 rejected options from the pick? | Higher = more decisive |
| **Pick distinctness (CER)** | Same, at character level | Higher = more decisive |
| **Confidence margin** | Score gap between 1st and 2nd pick | Higher = more confident |
| **Cross-lang σ(WER)** | Variation of proxy WER across languages | Lower = more consistent |
| **Selection distribution** | How often each option index is picked | Uniform = less bias |

#### Mode 2: Ground Truth Evaluation (with `--reference` or `reference` column)

If you have a labeled dataset (from Common Voice, FLEURS, MGB-2, or manual annotation), these are the real metrics:

| Metric | What it measures | Better |
|--------|-----------------|--------|
| **GT Accuracy** | % of times the scorer picked the option closest to the reference | Higher |
| **GT WER** | WER of the picked option vs the true reference | Lower |
| **GT CER** | CER of the picked option vs the true reference | Lower |
| **Oracle WER** | Best-possible WER (option closest to reference) | Lower |
| **WER gap** | GT WER - Oracle WER. How much room for improvement? | Lower = closer to optimal |

```bash
# With external reference CSV (columns: audio_id, reference)
python run_benchmark.py scorers.consensus_baseline.ConsensusBaseline \
    --data data/input.csv --reference data/references.csv

# Or add a 'reference' column directly to your data CSV — auto-detected
```

## Team Workflow

```
1. Everyone: git pull
2. Each person: cp scorers/template.py scorers/yourname.py
3. Implement your strategy
4. python run_benchmark.py scorers.yourname.YourScorer --data data/input.csv
5. git add benchmark_results/ && git commit
6. python compare_results.py → see who's winning
7. Iterate — adjust weights, swap models, try new signals
```

### Suggested Task Split (4 people)

| Person | Focus | Starter |
|--------|-------|---------|
| A | Whisper rescoring variants | `scorers/whisper_rescore.py` |
| B | CTC alignment + acoustic | `scorers/ctc_align.py` |
| C | LM fluency + text signals | `scorers/lm_perplexity.py` |
| D | Fusion ensemble tuning | `scorers/fusion_ensemble.py` |

## Tips

- Use `--limit 5` while developing to get fast feedback
- Use `--no-audio` for text-only approaches
- Audio is cached in `data/audio/` — first download is slow, then instant
- Check `benchmark_results/{name}_details.json` to debug specific samples
- The `setup()` method runs once — put all model loading there, not in `score()`
- The `language` parameter is passed per-sample — use it for multi-language support
- Lower `cross_language_consistency` = better generalization across languages
