# GoldenASR

Golden Transcription Selection Pipeline v5 for the Transcription Assessment challenge (Arabic\_SA dataset).

Given an audio sample and five candidate transcriptions, the pipeline identifies the **golden (ground-truth) transcription** using a two-regime adaptive ensemble of ASR similarity scoring, inter-option consensus analysis, and quality filtering.

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Repository Structure](#repository-structure)
3. [Quick Start](#quick-start)
4. [Module Reference](#module-reference)
5. [Notebook](#notebook)
6. [Results](#results)
7. [Report](#report)
8. [Team](#team)
9. [License](#license)

---

## Architecture Overview

| Component | Detail |
|---|---|
| ASR Model 1 | Whisper large-v3 (1.55B params, auto language detect, beam search) |
| ASR Model 2 | SeamlessM4T v2-large (~2.3B params, explicit `tgt_lang=arb`) |
| Hard Filters | Script detection (>1000 chars), speaker labels, stage directions, scene markers |
| Scoring Signals | ASR similarity, inter-option consensus, fluency, quality penalty, relative length |
| Regime Detection | Per-sample diversity - similar (consensus-heavy) vs diverse (ASR-heavy) |
| Optimization | Grid search over weight space, LOO-CV validation |

### Key Insight (Arabic\_SA)

All 49 labelled samples fall into the **similar regime** where options differ only in punctuation and formatting. The golden transcription is the "odd one out" with **negative consensus** (least similar to peers).

### Pipeline Flow

```
Audio + 5 Candidates
        |
        v
 +--------------+     +-----------------+
 | Whisper v3   |     | SeamlessM4T v2  |
 +--------------+     +-----------------+
        \                   /
         v                 v
     +-------------------------+
     | Signal Computation      |
     | (similarity, consensus, |
     |  fluency, quality)      |
     +-------------------------+
              |
              v
     +-------------------+
     | Regime Detection   |
     | (similar / diverse)|
     +-------------------+
              |
              v
     +--------------------+
     | Weighted Scoring    |
     | (per-regime weights)|
     +--------------------+
              |
              v
     +--------------------+
     | Golden Selection    |
     +--------------------+
```

---

## Repository Structure

```
GoldenASR/
|-- golden_asr/                  # Main Python package
|   |-- __init__.py              # Package metadata
|   |-- __main__.py              # CLI entry point
|   |-- config.py                # All configuration constants
|   |-- pipeline.py              # End-to-end pipeline orchestrator
|   |-- data/
|   |   |-- loader.py            # Dataset CSV loading and parsing
|   |   |-- downloader.py        # Parallel audio file downloading
|   |-- transcription/
|   |   |-- whisper_asr.py       # Whisper large-v3 backend
|   |   |-- seamless_asr.py      # SeamlessM4T v2-large backend
|   |-- preprocessing/
|   |   |-- normalization.py     # Arabic-optimized text normalization
|   |   |-- filters.py           # Hard quality filters
|   |-- scoring/
|   |   |-- signals.py           # Scoring signal computation
|   |   |-- regime.py            # Regime detection (similar/diverse)
|   |   |-- selection.py         # Weighted option selection
|   |-- optimization/
|   |   |-- grid_search.py       # Weight grid search
|   |   |-- validation.py        # LOO-CV and evaluation
|   |-- output/
|       |-- writer.py            # CSV output generation
|       |-- visualization.py     # Analysis panel plotting
|-- notebooks/
|   |-- golden_transcription_sa.ipynb   # Original monolithic notebook
|-- results/                     # Placeholder for output artifacts
|-- requirements.txt             # Python dependencies
|-- .gitignore
|-- README.md
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended for Whisper and SeamlessM4T inference)
- ffmpeg installed and on PATH

### Installation

```bash
git clone https://github.com/your-org/GoldenASR.git
cd GoldenASR
pip install -r requirements.txt
```

### Run the Pipeline

**As a Python module (CLI):**

```bash
python -m golden_asr --csv path/to/dataset.csv --output results/output.csv
```

**From Python code:**

```python
from golden_asr.pipeline import run

output_df = run(
    csv_path="path/to/dataset.csv",
    output_csv="results/output.csv",
    audio_dir="audio_files",
    plot_path="results/analysis.png",
)
```

**On Kaggle (auto-detect dataset):**

```python
from golden_asr.pipeline import run

output_df = run()  # CSV path auto-detected from /kaggle/input/
```

### CLI Options

| Flag | Description | Default |
|---|---|---|
| `--csv` | Path to dataset CSV | Auto-detected on Kaggle |
| `--output` | Path for submission CSV | `golden_transcriptions_output.csv` |
| `--audio-dir` | Directory for downloaded audio | `audio_files` |
| `--plot` | Path for analysis panel PNG | Derived from output path |

---

## Module Reference

### `golden_asr.config`
Central configuration file containing device settings, model parameters, grid search space, and default weights. Edit this file to tune hyperparameters without modifying pipeline logic.

### `golden_asr.data.loader`
Loads the transcription assessment CSV and parses the `correct_option` column into integer format. Supports auto-detection of dataset paths on Kaggle.

### `golden_asr.data.downloader`
Downloads audio files in parallel using a thread pool. Skips files that already exist locally.

### `golden_asr.transcription.whisper_asr`
Wraps OpenAI Whisper large-v3 for batch transcription. Handles model loading, inference with beam search and temperature fallback, and GPU memory cleanup.

### `golden_asr.transcription.seamless_asr`
Wraps Meta SeamlessM4T v2-large for batch transcription with explicit target language selection.

### `golden_asr.preprocessing.normalization`
Arabic-optimized text normalization: diacritic removal, alef variant normalization, punctuation stripping, and whitespace collapsing.

### `golden_asr.preprocessing.filters`
Hard quality filters that detect and reject candidates containing screenplay scripts, speaker labels, stage directions, or scene markers.

### `golden_asr.scoring.signals`
Computes per-option scoring signals: Whisper similarity, SeamlessM4T similarity, inter-option consensus, fluency proxy, quality penalty, and relative length.

### `golden_asr.scoring.regime`
Classifies each sample as "similar" or "diverse" based on option diversity (1 - mean consensus).

### `golden_asr.scoring.selection`
Scores candidate options using weighted signal combination and selects the best one, with regime-adaptive weight switching.

### `golden_asr.optimization.grid_search`
Exhaustive grid search over the six-dimensional weight space, run independently for each regime.

### `golden_asr.optimization.validation`
Evaluation utilities including weight testing on labeled data and leave-one-out cross-validation for model selection.

### `golden_asr.output.writer`
Generates the submission CSV with golden transcription predictions, per-option WER scores, and a detailed signal scores CSV.

### `golden_asr.output.visualization`
Produces a 2x3 analysis panel with golden option distribution, diversity histogram, regime accuracy comparison, weight visualization, signal comparison, and per-sample correctness scatter plot.

### `golden_asr.pipeline`
Orchestrates the full end-to-end pipeline: data loading, audio download, dual ASR transcription, signal computation, regime detection, grid search optimization, LOO-CV, prediction, output, and visualization.

---

## Notebook

The original monolithic Kaggle submission notebook is preserved at:

[notebooks/golden_transcription_sa.ipynb](notebooks/golden_transcription_sa.ipynb)

This notebook contains the same logic as the modularized package, structured for direct execution on Kaggle with GPU runtime. It is kept as a reference and for reproducibility of the original submission.

---

## Results

### Accuracy

| Metric | Value |
|---|---|
| Validation Accuracy | 47/49 (95.9%) |
| LOO-CV (Adaptive) | 47/49 (95.9%) |
| LOO-CV (Single) | 47/49 (95.9%) |
| Adaptive - Similar | 23/24 (95.8%) |
| Adaptive - Diverse | 24/25 (96.0%) |

### Optimized Weights

| Weight | Similar Regime | Diverse Regime | Single Regime |
|---|---|---|---|
| whisper | 0.0 | 0.0 | 0.0 |
| seamless | 0.0 | 0.0 | 0.0 |
| consensus | -0.1 | -0.1 | -0.1 |
| fluency | -0.1 | 0.0 | -0.1 |
| quality | 0.1 | 0.1 | 0.1 |
| length | 0.0 | 0.0 | 0.0 |

### Golden Option Distribution

| Option | Count |
|---|---|
| option_1 | 28 |
| option_2 | 24 |
| option_3 | 20 |
| option_4 | 15 |
| option_5 | 13 |

### Analysis Panel

<!-- PLACEHOLDER: Insert analysis_v5_arabic_sa.png here -->
<!-- ![Analysis Panel](results/analysis_v5_arabic_sa.png) -->

### Regime Distribution

| Regime | Count |
|---|---|
| Similar | 42 |
| Diverse | 58 |

**Diversity Threshold:** 0.0678  
**Mode Selected:** Adaptive two-regime  
**Dataset:** Arabic_SA (100 samples, 49 labeled)  
**ASR Models:** Whisper large-v3 + SeamlessM4T v2-large

---

## Report

<!-- PLACEHOLDER: Full report to be added in a subsequent commit -->
<!-- Include: methodology description, experimental setup, -->
<!-- ablation studies, error analysis, and conclusions -->

---

## Team

Team Rocket

---

## License


This project is provided for the Transcription Assessment challenge. See repository-level license for details.
