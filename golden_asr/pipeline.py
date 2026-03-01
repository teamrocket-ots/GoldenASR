"""
Main pipeline orchestrator for the GoldenASR golden transcription selection.

This module ties together all sub-modules into a single end-to-end pipeline:
  1. Load dataset
  2. Download audio
  3. Transcribe with Whisper and SeamlessM4T
  4. Compute scoring signals
  5. Detect regimes
  6. Optimize weights via grid search
  7. Validate with LOO-CV
  8. Generate predictions and output CSV
  9. Produce analysis visualizations
"""

import os

import numpy as np
import pandas as pd

from golden_asr.config import DIVERSITY_THRESHOLD, OUTPUT_CSV
from golden_asr.data.downloader import download_all_audio
from golden_asr.data.loader import auto_detect_csv, load_dataset
from golden_asr.optimization.grid_search import grid_search_regime, grid_search_single
from golden_asr.optimization.validation import evaluate_adaptive, loo_cv
from golden_asr.output.visualization import generate_analysis_panel
from golden_asr.output.writer import build_output_csv
from golden_asr.scoring.regime import auto_adjust_threshold, detect_regimes
from golden_asr.scoring.selection import pick_best_option, pick_best_option_adaptive
from golden_asr.scoring.signals import compute_all_signals
from golden_asr.transcription.seamless_asr import (
    transcribe_dataset as transcribe_seamless,
)
from golden_asr.transcription.whisper_asr import (
    transcribe_dataset as transcribe_whisper,
)


def run(csv_path=None, output_csv=None, audio_dir=None, plot_path=None):
    """Execute the full golden transcription selection pipeline.

    Args:
        csv_path: Path to the dataset CSV. If None, attempts auto-detection.
        output_csv: Path for the output submission CSV. Defaults to
            ``config.OUTPUT_CSV``.
        audio_dir: Directory for downloaded audio files.
        plot_path: Path for the analysis panel PNG.

    Returns:
        pd.DataFrame: The output DataFrame with golden transcription predictions.
    """
    output_csv = output_csv or OUTPUT_CSV
    diversity_threshold = DIVERSITY_THRESHOLD

    # ------------------------------------------------------------------ #
    # 1. Load dataset
    # ------------------------------------------------------------------ #
    if csv_path is None:
        csv_path = auto_detect_csv()
        if csv_path is None:
            csv_path = (
                "/kaggle/input/hackenza-3/"
                "Transcription Assessment Arabic_SA Dataset (1).csv"
            )
    print(f"CSV: {csv_path}")

    df = load_dataset(csv_path)
    labeled = df.dropna(subset=["correct_option_int"]).copy()
    labeled["correct_int"] = labeled["correct_option_int"].astype(int)
    print(
        f"Dataset: {len(df)} samples "
        f"({len(labeled)} labeled, {len(df) - len(labeled)} unlabeled)"
    )

    # ------------------------------------------------------------------ #
    # 2. Download audio
    # ------------------------------------------------------------------ #
    audio_paths = download_all_audio(df, audio_dir=audio_dir)
    df["audio_path"] = df["audio_id"].map(audio_paths)
    print(f"Audio files ready: {df['audio_path'].notna().sum()}/{len(df)}")

    # ------------------------------------------------------------------ #
    # 3. ASR transcription
    # ------------------------------------------------------------------ #
    whisper_texts, detected_langs = transcribe_whisper(df, audio_paths)
    df["whisper_text"] = df["audio_id"].map(whisper_texts)
    df["detected_lang"] = df["audio_id"].map(detected_langs)

    seamless_texts = transcribe_seamless(df, audio_paths)
    df["seamless_text"] = df["audio_id"].map(seamless_texts)

    # ------------------------------------------------------------------ #
    # 4. Scoring signals
    # ------------------------------------------------------------------ #
    signal_data = compute_all_signals(df, whisper_texts, seamless_texts)

    # ------------------------------------------------------------------ #
    # 5. Regime detection
    # ------------------------------------------------------------------ #
    sample_diversity, sample_regime = detect_regimes(
        signal_data, threshold=diversity_threshold
    )

    labeled["diversity"] = labeled["audio_id"].map(sample_diversity)
    labeled["regime"] = labeled["audio_id"].map(sample_regime)
    labeled_similar = labeled[labeled["regime"] == "similar"].copy()
    labeled_diverse = labeled[labeled["regime"] == "diverse"].copy()

    # Auto-adjust if one group is empty
    if len(labeled_similar) == 0 or len(labeled_diverse) == 0:
        diversity_threshold = auto_adjust_threshold(labeled, sample_diversity)
        labeled["regime"] = labeled["diversity"].apply(
            lambda d: "diverse" if d >= diversity_threshold else "similar"
        )
        sample_regime = {
            aid: ("diverse" if d >= diversity_threshold else "similar")
            for aid, d in sample_diversity.items()
        }
        labeled_similar = labeled[labeled["regime"] == "similar"].copy()
        labeled_diverse = labeled[labeled["regime"] == "diverse"].copy()
        print(f"Adjusted threshold: {diversity_threshold:.4f}")

    # ------------------------------------------------------------------ #
    # 6. Grid search
    # ------------------------------------------------------------------ #
    best_similar_weights, _, _ = grid_search_regime(
        labeled_similar, signal_data, "similar"
    )
    best_diverse_weights, _, _ = grid_search_regime(
        labeled_diverse, signal_data, "diverse"
    )
    best_single_weights, best_single_correct, best_single_acc = grid_search_single(
        labeled, signal_data
    )

    # Adaptive vs single comparison
    c, t, rc, rt = evaluate_adaptive(
        best_similar_weights,
        best_diverse_weights,
        labeled,
        signal_data,
        sample_diversity,
        diversity_threshold,
    )
    print(f"Adaptive: {c}/{t} = {c / t * 100:.1f}%")
    print(f"Single:   {best_single_correct}/{len(labeled)} = {best_single_acc * 100:.1f}%")

    # ------------------------------------------------------------------ #
    # 7. LOO-CV
    # ------------------------------------------------------------------ #
    cv_results = loo_cv(
        labeled,
        signal_data,
        sample_diversity,
        best_similar_weights,
        best_diverse_weights,
        best_single_weights,
        diversity_threshold,
    )

    use_adaptive = cv_results["adaptive_correct"] >= cv_results["single_correct"]
    mode = "Adaptive two-regime" if use_adaptive else "Single-regime"
    print(f"Selected mode: {mode}")

    # ------------------------------------------------------------------ #
    # 8. Generate predictions
    # ------------------------------------------------------------------ #
    final_predictions = {}
    for _, row in df.iterrows():
        aid = row["audio_id"]
        if aid not in signal_data:
            final_predictions[aid] = 1
            continue
        div = sample_diversity.get(aid, 0.0)
        if use_adaptive:
            final_predictions[aid] = pick_best_option_adaptive(
                signal_data[aid],
                best_similar_weights,
                best_diverse_weights,
                div,
                diversity_threshold,
            )
        else:
            final_predictions[aid] = pick_best_option(
                signal_data[aid], best_single_weights
            )

    regime_pred = {
        aid: ("diverse" if sample_diversity.get(aid, 0) >= diversity_threshold else "similar")
        for aid in final_predictions
    }

    # ------------------------------------------------------------------ #
    # 9. Output
    # ------------------------------------------------------------------ #
    output_df = build_output_csv(
        df, final_predictions, signal_data, sample_diversity, regime_pred, output_csv
    )

    # ------------------------------------------------------------------ #
    # 10. Visualization
    # ------------------------------------------------------------------ #
    if plot_path is None:
        plot_path = output_csv.replace(".csv", "_analysis.png")
    generate_analysis_panel(
        df,
        labeled,
        labeled_similar,
        labeled_diverse,
        signal_data,
        sample_diversity,
        final_predictions,
        best_similar_weights,
        best_diverse_weights,
        best_single_weights,
        diversity_threshold,
        plot_path,
    )

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    print("PIPELINE v5 - SUBMISSION SUMMARY")
    print("=" * 60)
    print(f"Dataset:             Arabic_SA ({len(df)} samples, {len(labeled)} labeled)")
    print(f"ASR Models:          Whisper large-v3 + SeamlessM4T v2-large")
    print(f"Mode:                {mode}")
    print(f"Diversity Threshold: {diversity_threshold:.4f}")
    print(f"LOO-CV (adaptive):   {cv_results['adaptive_correct']}/{cv_results['total']}")
    print(f"LOO-CV (single):     {cv_results['single_correct']}/{cv_results['total']}")
    print(f"Total predictions:   {len(final_predictions)}")
    print(f"Output CSV:          {output_csv}")
    print("=" * 60)
    print("SUBMISSION READY")
    print("=" * 60)

    return output_df
