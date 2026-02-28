"""
Benchmark Runner — loads data, runs your scorer, computes metrics, saves results.

Usage:
    from benchmark.runner import run_benchmark
    from scorers.my_scorer import MyScorer
    run_benchmark(MyScorer(), "data/input.csv")

Or from CLI:
    python run_benchmark.py scorers.my_scorer.MyScorer --data data/input.csv
"""

import os
import json
import time
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
from tqdm import tqdm
from datetime import datetime

from benchmark.base_scorer import BaseScorer
from golden_asr.utils.wer import compute_wer, compute_cer
from golden_asr.scorers.consensus_mbr import ConsensusMBRScorer


# ── Data Loading (standalone, no Config dependency) ──────────────────


def _load_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    option_cols = [c for c in df.columns if c.startswith("option_")]
    for c in option_cols:
        df[c] = df[c].fillna("")
    return df


def _get_options(row: pd.Series) -> List[str]:
    return [str(row[f"option_{i}"]) for i in range(1, 6)]


def _download_audio(df: pd.DataFrame, audio_dir: str) -> Dict[str, str]:
    """Download audio files, return {audio_id: local_path}. Skips existing."""
    import requests
    os.makedirs(audio_dir, exist_ok=True)
    audio_map = {}
    for _, row in df.iterrows():
        aid = str(row["audio_id"])
        url = str(row.get("audio", ""))
        if not url or url == "nan":
            continue
        ext = os.path.splitext(url.split("?")[0])[-1] or ".wav"
        path = os.path.join(audio_dir, f"{aid}{ext}")
        audio_map[aid] = path
        if os.path.exists(path):
            continue
        try:
            r = requests.get(url, timeout=60, stream=True)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        except Exception as e:
            print(f"  [download] Failed {aid}: {e}")
    return audio_map


# ── Metrics ──────────────────────────────────────────────────────────


def _compute_metrics(
    selections: List[dict], df: pd.DataFrame, ref_map: Optional[Dict[str, str]] = None
) -> dict:
    """
    Compute all benchmark metrics from selection results.

    Two evaluation modes:
      1. WITHOUT ground truth (ref_map=None): self-referential proxy metrics
         - How distinct is the pick from the other 4?
         - Does this scorer agree with MBR consensus?
      2. WITH ground truth (ref_map provided): real accuracy metrics
         - WER/CER of the selected option vs the true reference
         - Accuracy: did the scorer pick the closest option to truth?
    """

    n = len(selections)
    if n == 0:
        return {"error": "no samples processed"}

    has_refs = ref_map is not None and len(ref_map) > 0

    # --- Basic tracking ---
    golden_indices = []
    times = []

    for s in selections:
        golden_indices.append(s["golden_idx"])
        times.append(s["time_s"])

    # --- Self-referential proxy metrics (always computed) ---
    # These measure "how different is the pick from the other options"
    # NOT accuracy. Useful for comparing scorers when no ground truth exists.
    proxy_wer = []
    proxy_cer = []
    for s in selections:
        golden = s["options"][s["golden_idx"]]
        for i, opt in enumerate(s["options"]):
            if i != s["golden_idx"]:
                proxy_wer.append(compute_wer(golden, opt))
                proxy_cer.append(compute_cer(golden, opt))

    # --- Agreement with consensus MBR baseline ---
    mbr = ConsensusMBRScorer()
    agree_count = 0
    for s in selections:
        mbr_scores = mbr.score_all(s["options"])
        mbr_pick = int(np.argmax(mbr_scores))
        if mbr_pick == s["golden_idx"]:
            agree_count += 1
    agreement = agree_count / n

    # --- Selection distribution ---
    from collections import Counter
    dist = Counter(f"option_{i+1}" for i in golden_indices)

    # --- Per-language breakdown ---
    lang_data: Dict[str, list] = {}
    for s in selections:
        lang = s.get("language", "unknown")
        lang_data.setdefault(lang, []).append(s)

    per_lang = {}
    for lang, samples in lang_data.items():
        lang_times = [s["time_s"] for s in samples]
        lang_proxy_wer = []
        lang_proxy_cer = []
        for s in samples:
            golden = s["options"][s["golden_idx"]]
            for i, opt in enumerate(s["options"]):
                if i != s["golden_idx"]:
                    lang_proxy_wer.append(compute_wer(golden, opt))
                    lang_proxy_cer.append(compute_cer(golden, opt))
        lang_agree = 0
        for s in samples:
            mbr_scores = mbr.score_all(s["options"])
            if int(np.argmax(mbr_scores)) == s["golden_idx"]:
                lang_agree += 1
        lang_info = {
            "count": len(samples),
            "avg_time_s": round(np.mean(lang_times), 4),
            "proxy_wer_distinctness": round(np.mean(lang_proxy_wer), 4) if lang_proxy_wer else 0,
            "proxy_cer_distinctness": round(np.mean(lang_proxy_cer), 4) if lang_proxy_cer else 0,
            "mbr_agreement": round(lang_agree / len(samples), 4),
        }
        # Ground truth per-language metrics
        if has_refs:
            lang_gt_wer = []
            lang_gt_cer = []
            lang_correct = 0
            for s in samples:
                ref = ref_map.get(s["audio_id"], "")
                if not ref:
                    continue
                picked = s["options"][s["golden_idx"]]
                lang_gt_wer.append(compute_wer(ref, picked))
                lang_gt_cer.append(compute_cer(ref, picked))
                # Check if scorer picked the option with lowest WER to ref
                option_wers = [compute_wer(ref, opt) for opt in s["options"]]
                best_idx = int(np.argmin(option_wers))
                if s["golden_idx"] == best_idx:
                    lang_correct += 1
            if lang_gt_wer:
                lang_info["gt_wer"] = round(np.mean(lang_gt_wer), 4)
                lang_info["gt_cer"] = round(np.mean(lang_gt_cer), 4)
                lang_info["gt_accuracy"] = round(lang_correct / len(lang_gt_wer), 4)

        per_lang[lang] = lang_info

    # --- Cross-language consistency ---
    proxy_key = "gt_wer" if has_refs else "proxy_wer_distinctness"
    lang_vals = [v[proxy_key] for v in per_lang.values() if proxy_key in v and v["count"] >= 2]
    cross_lang_consistency = round(float(np.std(lang_vals)), 4) if len(lang_vals) >= 2 else 0.0

    # --- Per-option divergence from pick ---
    option_vs_pick_wer = {}
    for opt_i in range(5):
        key = f"avg_wer_option{opt_i+1}_vs_pick"
        wers = []
        for s in selections:
            picked = s["options"][s["golden_idx"]]
            wers.append(compute_wer(picked, s["options"][opt_i]))
        option_vs_pick_wer[key] = round(np.mean(wers), 4)

    # --- Score confidence (gap between 1st and 2nd best) ---
    confidence_margins = []
    for s in selections:
        sorted_scores = sorted(s["scores"], reverse=True)
        if len(sorted_scores) >= 2:
            margin = sorted_scores[0] - sorted_scores[1]
            confidence_margins.append(margin)
    avg_confidence = round(np.mean(confidence_margins), 4) if confidence_margins else 0.0

    metrics = {
        "total_samples": n,
        "has_ground_truth": has_refs,
        "avg_time_per_sample_s": round(np.mean(times), 4),
        "total_time_s": round(sum(times), 2),
        # Proxy metrics (always available — self-referential, NOT accuracy)
        "mbr_agreement": round(agreement, 4),
        "proxy_wer_distinctness": round(np.mean(proxy_wer), 4) if proxy_wer else 0,
        "proxy_cer_distinctness": round(np.mean(proxy_cer), 4) if proxy_cer else 0,
        "avg_confidence_margin": avg_confidence,
        "cross_language_consistency": cross_lang_consistency,
        "selection_distribution": dict(dist),
        "per_language": per_lang,
        **option_vs_pick_wer,
    }

    # --- Ground truth metrics (only when reference labels exist) ---
    if has_refs:
        gt_wers = []
        gt_cers = []
        correct_picks = 0
        total_with_ref = 0
        oracle_wers = []  # best-possible WER (option closest to ref)

        for s in selections:
            ref = ref_map.get(s["audio_id"], "")
            if not ref:
                continue
            total_with_ref += 1
            picked = s["options"][s["golden_idx"]]
            gt_wers.append(compute_wer(ref, picked))
            gt_cers.append(compute_cer(ref, picked))

            # Did the scorer pick the best option?
            option_wers = [compute_wer(ref, opt) for opt in s["options"]]
            best_idx = int(np.argmin(option_wers))
            oracle_wers.append(option_wers[best_idx])
            if s["golden_idx"] == best_idx:
                correct_picks += 1

        if total_with_ref > 0:
            metrics["gt_samples"] = total_with_ref
            metrics["gt_wer"] = round(np.mean(gt_wers), 4)
            metrics["gt_cer"] = round(np.mean(gt_cers), 4)
            metrics["gt_accuracy"] = round(correct_picks / total_with_ref, 4)
            metrics["gt_oracle_wer"] = round(np.mean(oracle_wers), 4)
            metrics["gt_wer_gap"] = round(metrics["gt_wer"] - metrics["gt_oracle_wer"], 4)

    return metrics


# ── Main Runner ──────────────────────────────────────────────────────


def run_benchmark(
    scorer: BaseScorer,
    data_csv: str = "data/input.csv",
    audio_dir: str = "data/audio",
    output_dir: str = "benchmark_results",
    download_audio: bool = True,
    limit: Optional[int] = None,
    reference_csv: Optional[str] = None,
) -> dict:
    """
    Run a scorer against the dataset and produce metrics + output files.

    Args:
        scorer:         Your BaseScorer implementation
        data_csv:       Path to input CSV
        audio_dir:      Where to cache downloaded audio
        output_dir:     Where to save results
        download_audio: Whether to download audio files
        limit:          Process only first N samples (for quick testing)
        reference_csv:  Optional CSV with columns: audio_id, reference
                        If provided, computes real WER/accuracy against ground truth.
                        If your data CSV already has a 'reference' column, that
                        is used automatically (no separate file needed).

    Returns:
        dict with all metrics
    """
    print(f"\n{'='*60}")
    print(f"  BENCHMARK: {scorer.name}")
    print(f"  Data:      {data_csv}")
    print(f"{'='*60}\n")

    # Load data
    df = _load_csv(data_csv)
    if limit:
        df = df.head(limit)
    print(f"Loaded {len(df)} samples")

    # Download audio
    audio_map = {}
    if download_audio and "audio" in df.columns:
        print("Downloading audio...")
        audio_map = _download_audio(df, audio_dir)
        print(f"Audio ready: {len(audio_map)} files\n")

    # Load ground-truth references (if available)
    ref_map: Optional[Dict[str, str]] = None
    if reference_csv:
        ref_df = pd.read_csv(reference_csv)
        ref_map = dict(zip(ref_df["audio_id"].astype(str), ref_df["reference"].astype(str)))
        print(f"Loaded {len(ref_map)} ground-truth references from {reference_csv}")
    elif "reference" in df.columns:
        ref_map = {}
        for _, row in df.iterrows():
            ref_text = str(row["reference"]).strip()
            if ref_text and ref_text != "nan":
                ref_map[str(row["audio_id"])] = ref_text
        if ref_map:
            print(f"Found {len(ref_map)} ground-truth references in data CSV")
        else:
            ref_map = None

    eval_mode = "GROUND TRUTH" if ref_map else "PROXY (no ground truth)"
    print(f"Evaluation mode: {eval_mode}\n")

    # Setup scorer
    print(f"Setting up scorer '{scorer.name}'...")
    scorer.setup()

    # Run scoring
    selections = []
    rows_out = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Scoring [{scorer.name}]"):
        audio_id = str(row["audio_id"])
        options = _get_options(row)
        language = str(row.get("language", "unknown"))
        audio_path = audio_map.get(audio_id)

        t0 = time.time()
        try:
            scores = scorer.score(options, audio_path=audio_path, language=language)
        except Exception as e:
            print(f"  [ERROR] {audio_id}: {e}")
            scores = [0.0] * 5
        elapsed = time.time() - t0

        golden_idx = int(np.argmax(scores))
        golden_text = options[golden_idx]

        selections.append({
            "audio_id": audio_id,
            "language": language,
            "options": options,
            "scores": scores,
            "golden_idx": golden_idx,
            "time_s": elapsed,
        })

        # Build output row
        out = {
            "audio_id": audio_id,
            "language": language,
            "audio": row.get("audio", ""),
        }
        for i, opt in enumerate(options):
            out[f"option_{i+1}"] = opt
        out["golden_ref"] = golden_text
        out["golden_option"] = f"option_{golden_idx + 1}"
        out["golden_score"] = round(scores[golden_idx], 6)
        for i, opt in enumerate(options):
            out[f"wer_option{i+1}"] = round(compute_wer(golden_text, opt), 4)

        # Ground truth columns (when available)
        if ref_map and audio_id in ref_map:
            ref_text = ref_map[audio_id]
            out["reference"] = ref_text
            out["gt_wer"] = round(compute_wer(ref_text, golden_text), 4)
            out["gt_cer"] = round(compute_cer(ref_text, golden_text), 4)
            # Which option was actually closest to the reference?
            option_wers = [compute_wer(ref_text, opt) for opt in options]
            oracle_idx = int(np.argmin(option_wers))
            out["oracle_option"] = f"option_{oracle_idx + 1}"
            out["oracle_wer"] = round(option_wers[oracle_idx], 4)
            out["correct"] = int(golden_idx == oracle_idx)

        rows_out.append(out)

    # Teardown
    scorer.teardown()

    # Compute metrics
    metrics = _compute_metrics(selections, df, ref_map=ref_map)
    metrics["scorer_name"] = scorer.name
    metrics["data_csv"] = data_csv
    metrics["timestamp"] = datetime.now().isoformat()

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)
    safe_name = scorer.name.replace("/", "_").replace(" ", "_")

    # 1. Output CSV (submission format)
    csv_path = os.path.join(output_dir, f"{safe_name}_results.csv")
    pd.DataFrame(rows_out).to_csv(csv_path, index=False)

    # 2. Metrics JSON
    json_path = os.path.join(output_dir, f"{safe_name}_metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # 3. Per-sample scores (for debugging)
    detail_path = os.path.join(output_dir, f"{safe_name}_details.json")
    with open(detail_path, "w") as f:
        json.dump(selections, f, indent=2, ensure_ascii=False, default=str)

    # Print summary
    print(f"\n{'='*60}")
    print(f"  RESULTS: {scorer.name}")
    print(f"{'='*60}")
    print(f"  Samples:              {metrics['total_samples']}")
    print(f"  Total time:           {metrics['total_time_s']:.1f}s")
    print(f"  Avg time/sample:      {metrics['avg_time_per_sample_s']:.4f}s")

    if metrics.get("has_ground_truth"):
        print(f"\n  ── Ground Truth Evaluation ──")
        print(f"  GT Accuracy:          {metrics.get('gt_accuracy', 0):.1%}")
        print(f"  GT WER (picked):      {metrics.get('gt_wer', 0):.4f}")
        print(f"  GT CER (picked):      {metrics.get('gt_cer', 0):.4f}")
        print(f"  Oracle WER (best):    {metrics.get('gt_oracle_wer', 0):.4f}")
        print(f"  WER gap (you-oracle): {metrics.get('gt_wer_gap', 0):.4f}")

    print(f"\n  ── Proxy Metrics (no ground truth needed) ──")
    print(f"  MBR agreement:        {metrics['mbr_agreement']:.1%}")
    print(f"  Pick distinctness:    WER {metrics['proxy_wer_distinctness']:.4f}, CER {metrics['proxy_cer_distinctness']:.4f}")
    print(f"  Confidence margin:    {metrics['avg_confidence_margin']:.4f}")
    if metrics.get("cross_language_consistency", 0) > 0:
        print(f"  Cross-lang σ(WER):    {metrics['cross_language_consistency']:.4f}")
    print(f"\n  Selection distribution:")
    for opt, count in sorted(metrics["selection_distribution"].items()):
        pct = count / metrics["total_samples"] * 100
        print(f"    {opt}: {count} ({pct:.1f}%)")
    if metrics.get("per_language"):
        print(f"\n  Per-language breakdown:")
        for lang, info in sorted(metrics["per_language"].items()):
            parts = [f"{info['count']} samples"]
            if "gt_wer" in info:
                parts.append(f"GT-WER {info['gt_wer']:.4f}")
                parts.append(f"ACC {info.get('gt_accuracy', 0):.1%}")
            else:
                parts.append(f"proxy-WER {info.get('proxy_wer_distinctness', 0):.4f}")
            parts.append(f"MBR {info.get('mbr_agreement', 0):.1%}")
            parts.append(f"{info['avg_time_s']:.4f}s/sample")
            print(f"    {lang}: {', '.join(parts)}")
    print(f"\n  Output files:")
    print(f"    CSV:     {csv_path}")
    print(f"    Metrics: {json_path}")
    print(f"    Details: {detail_path}")
    print(f"{'='*60}\n")

    return metrics
