"""
Regime detection based on per-sample option diversity.

Diversity is defined as ``1 - mean_consensus``. Low diversity indicates
similar options (differences only in punctuation/formatting), while high
diversity indicates substantively different candidates.
"""

import numpy as np
import pandas as pd

from golden_asr.config import DIVERSITY_THRESHOLD


def compute_sample_diversity(row_signals):
    """Compute diversity for a single sample.

    Args:
        row_signals: Dict mapping option number to signal dict.

    Returns:
        float: Diversity score in [0, 1].
    """
    valid_keys = [j for j, s in row_signals.items() if not s.get("is_script", False)]
    if len(valid_keys) < 2:
        return 0.0
    consensuses = [row_signals[j].get("consensus", 0.5) for j in valid_keys]
    return 1.0 - np.mean(consensuses)


def detect_regimes(signal_data, threshold=None):
    """Classify every sample as 'similar' or 'diverse'.

    Args:
        signal_data: Dict mapping audio_id to signal dicts.
        threshold: Diversity threshold. Defaults to ``config.DIVERSITY_THRESHOLD``.

    Returns:
        tuple: (sample_diversity dict, sample_regime dict).
    """
    threshold = threshold if threshold is not None else DIVERSITY_THRESHOLD

    sample_diversity = {
        aid: compute_sample_diversity(sigs) for aid, sigs in signal_data.items()
    }
    sample_regime = {
        aid: ("diverse" if d >= threshold else "similar")
        for aid, d in sample_diversity.items()
    }

    div_values = list(sample_diversity.values())
    regime_counts = pd.Series(sample_regime).value_counts()

    print("=" * 60)
    print("REGIME DETECTION")
    print("=" * 60)
    print(f"Threshold: {threshold}")
    print(
        f"Diversity: mean={np.mean(div_values):.3f}, "
        f"min={np.min(div_values):.3f}, max={np.max(div_values):.3f}"
    )
    print(
        f"Similar: {regime_counts.get('similar', 0)}, "
        f"Diverse: {regime_counts.get('diverse', 0)}"
    )

    return sample_diversity, sample_regime


def auto_adjust_threshold(labeled_df, sample_diversity):
    """Adjust the threshold to the median if one regime group is empty.

    Args:
        labeled_df: Labeled subset of the dataset (must have ``audio_id``).
        sample_diversity: Dict mapping audio_id to diversity scores.

    Returns:
        float: Adjusted threshold value.
    """
    diversities = [sample_diversity.get(aid, 0.0) for aid in labeled_df["audio_id"]]
    return float(np.median(diversities))
