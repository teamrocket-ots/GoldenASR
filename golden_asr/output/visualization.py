"""
Visualization utilities for analysis and reporting.
"""

import numpy as np
import pandas as pd

from golden_asr.scoring.selection import pick_best_option, pick_best_option_adaptive


def generate_analysis_panel(
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
    output_path,
):
    """Generate a 2x3 analysis panel and save it as a PNG.

    Args:
        df: Full dataset DataFrame.
        labeled: Labeled subset DataFrame.
        labeled_similar: Similar-regime labeled subset.
        labeled_diverse: Diverse-regime labeled subset.
        signal_data: Full signal data dict.
        sample_diversity: Dict mapping audio_id to diversity.
        final_predictions: Dict mapping audio_id to predicted option.
        best_similar_weights: Optimized similar-regime weights.
        best_diverse_weights: Optimized diverse-regime weights.
        best_single_weights: Optimized single-regime weights.
        diversity_threshold: Regime split threshold.
        output_path: Path to save the output PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(
        "Golden Transcription Selection v5 - Arabic_SA",
        fontsize=15,
        fontweight="bold",
    )

    # (a) Golden option distribution
    ax = axes[0, 0]
    df["golden_option"] = df["audio_id"].map(final_predictions)
    df["golden_option"].value_counts().sort_index().plot(
        kind="bar", ax=ax, color="steelblue"
    )
    ax.set_title("(a) Golden Option Distribution")
    ax.set_xlabel("Option")
    ax.set_ylabel("Count")

    # (b) Diversity histogram
    ax = axes[0, 1]
    divs = [sample_diversity.get(aid, 0) for aid in df["audio_id"]]
    ax.hist(divs, bins=20, color="mediumpurple", edgecolor="black", alpha=0.7)
    ax.axvline(
        diversity_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Threshold={diversity_threshold:.3f}",
    )
    ax.set_title("(b) Diversity Distribution")
    ax.set_xlabel("Diversity")
    ax.legend()

    # (c) Per-regime accuracy
    ax = axes[0, 2]
    regimes_list = ["Similar", "Diverse", "Overall"]
    adap_accs = []
    single_accs = []
    for sub_df in [labeled_similar, labeled_diverse, labeled]:
        if len(sub_df) > 0:
            ca = cs = 0
            for _, r in sub_df.iterrows():
                aid = r["audio_id"]
                actual = int(r["correct_int"])
                d = sample_diversity.get(aid, 0)
                if (
                    pick_best_option_adaptive(
                        signal_data[aid],
                        best_similar_weights,
                        best_diverse_weights,
                        d,
                        diversity_threshold,
                    )
                    == actual
                ):
                    ca += 1
                if pick_best_option(signal_data[aid], best_single_weights) == actual:
                    cs += 1
            adap_accs.append(ca / len(sub_df) * 100)
            single_accs.append(cs / len(sub_df) * 100)
        else:
            adap_accs.append(0)
            single_accs.append(0)
    x = np.arange(len(regimes_list))
    ax.bar(x - 0.2, adap_accs, 0.35, label="Adaptive", color="green", alpha=0.7)
    ax.bar(x + 0.2, single_accs, 0.35, label="Single", color="orange", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes_list)
    ax.set_title("(c) Adaptive vs Single")
    ax.set_ylabel("Accuracy %")
    ax.legend()

    # (d) Weight comparison
    ax = axes[1, 0]
    w_names = list(best_similar_weights.keys())
    x = np.arange(len(w_names))
    ax.bar(
        x - 0.2,
        [best_similar_weights[k] for k in w_names],
        0.35,
        label="Similar",
        color="skyblue",
    )
    ax.bar(
        x + 0.2,
        [best_diverse_weights[k] for k in w_names],
        0.35,
        label="Diverse",
        color="coral",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(w_names, rotation=30, fontsize=9)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("(d) Weights: Similar vs Diverse")
    ax.legend()

    # (e) Signal comparison: golden vs non-golden
    ax = axes[1, 1]
    detail_rows = []
    for aid, sigs in signal_data.items():
        for j, s in sigs.items():
            if s.get("is_script", False):
                continue
            detail_rows.append(
                {
                    "whisper_sim": s.get("whisper_sim", 0),
                    "seamless_sim": s.get("seamless_sim", 0),
                    "consensus": s.get("consensus", 0),
                    "quality_penalty": s.get("quality_penalty", 0),
                    "is_golden": j == final_predictions.get(aid, -1),
                }
            )
    detail_df = pd.DataFrame(detail_rows)
    sig_names = ["whisper_sim", "seamless_sim", "consensus", "quality_penalty"]
    golden_m = [detail_df[detail_df["is_golden"]][s].mean() for s in sig_names]
    nongolden_m = [detail_df[~detail_df["is_golden"]][s].mean() for s in sig_names]
    x = np.arange(len(sig_names))
    ax.bar(x - 0.2, golden_m, 0.35, label="Golden", color="green", alpha=0.7)
    ax.bar(x + 0.2, nongolden_m, 0.35, label="Non-Golden", color="red", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(sig_names, rotation=30, fontsize=9)
    ax.set_title("(e) Signal: Golden vs Non-Golden")
    ax.legend()

    # (f) Diversity vs correctness
    ax = axes[1, 2]
    for idx, (_, r) in enumerate(labeled.iterrows()):
        aid = r["audio_id"]
        d = sample_diversity.get(aid, 0)
        correct = final_predictions.get(aid) == int(r["correct_int"])
        ax.scatter(
            d,
            idx,
            c="green" if correct else "red",
            marker="o" if d < diversity_threshold else "s",
            s=40,
            alpha=0.7,
        )
    ax.axvline(diversity_threshold, color="blue", linestyle="--", alpha=0.5)
    ax.set_xlabel("Diversity")
    ax.set_ylabel("Sample Index")
    ax.set_title("(f) Diversity vs Correctness")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Analysis panel saved: {output_path}")
