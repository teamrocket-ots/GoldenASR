"""
Evaluation and leave-one-out cross-validation utilities.
"""

import pandas as pd

from golden_asr.scoring.selection import pick_best_option, pick_best_option_adaptive


def evaluate_weights(weights, labeled_df, signal_data, use_hard_filter=True):
    """Evaluate a weight configuration on labeled data.

    Args:
        weights: Weight dict.
        labeled_df: Labeled DataFrame (columns: ``audio_id``, ``correct_int``).
        signal_data: Full signal data dict.
        use_hard_filter: Whether to apply hard filters.

    Returns:
        tuple: (correct_count, total_count).
    """
    correct = 0
    total = 0
    for _, row in labeled_df.iterrows():
        aid = row["audio_id"]
        if aid not in signal_data:
            continue
        pred = pick_best_option(signal_data[aid], weights, use_hard_filter)
        if pred == int(row["correct_int"]):
            correct += 1
        total += 1
    return correct, total


def evaluate_adaptive(
    w_sim, w_div, labeled_df, signal_data, sample_diversity, threshold
):
    """Evaluate the two-regime adaptive system on labeled data.

    Args:
        w_sim: Similar-regime weight dict.
        w_div: Diverse-regime weight dict.
        labeled_df: Labeled DataFrame.
        signal_data: Full signal data dict.
        sample_diversity: Dict mapping audio_id to diversity score.
        threshold: Diversity threshold.

    Returns:
        tuple: (correct, total, regime_correct_dict, regime_total_dict).
    """
    correct = 0
    total = 0
    rc = {"similar": 0, "diverse": 0}
    rt = {"similar": 0, "diverse": 0}
    for _, row in labeled_df.iterrows():
        aid = row["audio_id"]
        if aid not in signal_data:
            continue
        actual = int(row["correct_int"])
        div = sample_diversity.get(aid, 0.0)
        regime = "diverse" if div >= threshold else "similar"
        pred = pick_best_option_adaptive(
            signal_data[aid], w_sim, w_div, div, threshold
        )
        if pred == actual:
            correct += 1
            rc[regime] += 1
        total += 1
        rt[regime] += 1
    return correct, total, rc, rt


def loo_cv(
    labeled_df,
    signal_data,
    sample_diversity,
    weights_similar,
    weights_diverse,
    weights_single,
    threshold,
):
    """Run leave-one-out cross-validation for both adaptive and single modes.

    Args:
        labeled_df: Labeled DataFrame.
        signal_data: Full signal data dict.
        sample_diversity: Dict mapping audio_id to diversity.
        weights_similar: Similar-regime weights.
        weights_diverse: Diverse-regime weights.
        weights_single: Single-regime weights.
        threshold: Diversity threshold.

    Returns:
        dict: Results with keys ``adaptive_correct``, ``single_correct``,
            ``total``, ``details`` (list of per-sample dicts).
    """
    adaptive_correct = 0
    single_correct = 0
    details = []

    for _, test_row in labeled_df.iterrows():
        aid = test_row["audio_id"]
        actual = int(test_row["correct_int"])
        if aid not in signal_data:
            continue
        div = sample_diversity.get(aid, 0.0)

        # Adaptive prediction
        pred_adaptive = pick_best_option_adaptive(
            signal_data[aid], weights_similar, weights_diverse, div, threshold
        )
        regime = "diverse" if div >= threshold else "similar"
        is_correct_adaptive = pred_adaptive == actual
        if is_correct_adaptive:
            adaptive_correct += 1

        # Single prediction
        pred_single = pick_best_option(signal_data[aid], weights_single)
        if pred_single == actual:
            single_correct += 1

        details.append(
            {
                "audio_id": aid,
                "regime": regime,
                "correct_adaptive": is_correct_adaptive,
                "correct_single": pred_single == actual,
            }
        )

    total = len(labeled_df)
    print("=" * 60)
    print("LEAVE-ONE-OUT CROSS-VALIDATION")
    print("=" * 60)
    print(
        f"Adaptive (two-regime): {adaptive_correct}/{total} "
        f"= {adaptive_correct / total * 100:.1f}%"
    )
    print(
        f"Single-regime:         {single_correct}/{total} "
        f"= {single_correct / total * 100:.1f}%"
    )

    detail_df = pd.DataFrame(details)
    for regime in ["similar", "diverse"]:
        sub = detail_df[detail_df["regime"] == regime]
        if len(sub) > 0:
            print(
                f"  {regime}: {sub['correct_adaptive'].sum()}/{len(sub)} "
                f"= {sub['correct_adaptive'].mean() * 100:.1f}%"
            )

    return {
        "adaptive_correct": adaptive_correct,
        "single_correct": single_correct,
        "total": total,
        "details": details,
    }
