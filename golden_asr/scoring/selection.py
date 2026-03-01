"""
Option selection using weighted signal scoring.
"""


def pick_best_option(row_signals, weights, use_hard_filter=True):
    """Score candidate options and return the best one.

    Args:
        row_signals: Dict mapping option number to signal dict.
        weights: Dict with keys ``whisper``, ``seamless``, ``consensus``,
            ``fluency``, ``quality``, ``length``.
        use_hard_filter: If True, skip hard-filtered options. On fallback
            (no candidates remain), retries without the filter.

    Returns:
        int: The best option number.
    """
    candidates = {}
    for j, sig in row_signals.items():
        if sig.get("is_script", False):
            continue
        if use_hard_filter and sig.get("hard_filter", False):
            continue
        score = (
            weights.get("whisper", 0) * sig.get("whisper_sim", 0)
            + weights.get("seamless", 0) * sig.get("seamless_sim", 0)
            + weights.get("consensus", 0) * sig.get("consensus", 0)
            + weights.get("fluency", 0) * sig.get("fluency", 0)
            - weights.get("quality", 0) * sig.get("quality_penalty", 0)
            - weights.get("length", 0) * sig.get("rel_length", 0.5)
        )
        candidates[j] = score
    if not candidates and use_hard_filter:
        return pick_best_option(row_signals, weights, use_hard_filter=False)
    if not candidates:
        return 1
    return max(candidates, key=candidates.get)


def pick_best_option_adaptive(
    row_signals, weights_similar, weights_diverse, diversity, threshold,
    use_hard_filter=True,
):
    """Select the best option using regime-adaptive weights.

    Chooses between ``weights_similar`` and ``weights_diverse`` based on
    whether the sample's diversity exceeds the threshold.

    Args:
        row_signals: Dict mapping option number to signal dict.
        weights_similar: Weight dict for the similar regime.
        weights_diverse: Weight dict for the diverse regime.
        diversity: Sample diversity score.
        threshold: Regime split threshold.
        use_hard_filter: Whether to apply hard filters.

    Returns:
        int: The best option number.
    """
    w = weights_diverse if diversity >= threshold else weights_similar
    return pick_best_option(row_signals, w, use_hard_filter)
