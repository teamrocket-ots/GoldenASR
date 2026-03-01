"""
Grid search over the weight space for both similar and diverse regimes.
"""

from itertools import product as iter_product

from tqdm.auto import tqdm

from golden_asr.config import DEFAULT_DIVERSE_WEIGHTS, DEFAULT_SIMILAR_WEIGHTS, GRID
from golden_asr.optimization.validation import evaluate_weights


def _total_combos(grid):
    total = 1
    for v in grid.values():
        total *= len(v)
    return total


def grid_search_single(labeled_df, signal_data, grid=None):
    """Run a grid search over the full labeled set (single-regime baseline).

    Args:
        labeled_df: Labeled DataFrame (must have ``audio_id``, ``correct_int``).
        signal_data: Full signal data dict.
        grid: Weight grid dict. Defaults to ``config.GRID``.

    Returns:
        tuple: (best_weights, best_correct, best_accuracy).
    """
    grid = grid or GRID
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    total = _total_combos(grid)

    best_acc = 0
    best_weights = None
    best_correct = 0

    for combo in tqdm(iter_product(*value_lists), total=total, desc="Single grid"):
        weights = dict(zip(keys, combo))
        if weights["whisper"] == 0 and weights["seamless"] == 0 and weights["consensus"] == 0:
            continue
        c, t = evaluate_weights(weights, labeled_df, signal_data)
        acc = c / t if t > 0 else 0
        if acc > best_acc or (
            acc == best_acc
            and sum(abs(v) for v in weights.values())
            < sum(abs(v) for v in (best_weights or weights).values())
        ):
            best_acc = acc
            best_weights = weights.copy()
            best_correct = c

    print(f"Best SINGLE: {best_correct}/{len(labeled_df)} = {best_acc * 100:.1f}%")
    print(f"Weights: {best_weights}")
    return best_weights, best_correct, best_acc


def grid_search_regime(labeled_df, signal_data, regime_name, grid=None):
    """Run a grid search for a specific regime subset.

    Args:
        labeled_df: Labeled DataFrame for this regime only.
        signal_data: Full signal data dict.
        regime_name: 'similar' or 'diverse' (for logging).
        grid: Weight grid dict. Defaults to ``config.GRID``.

    Returns:
        tuple: (best_weights, best_correct, best_accuracy).
    """
    grid = grid or GRID
    keys = list(grid.keys())
    value_lists = [grid[k] for k in keys]
    total = _total_combos(grid)

    defaults = {
        "similar": DEFAULT_SIMILAR_WEIGHTS,
        "diverse": DEFAULT_DIVERSE_WEIGHTS,
    }

    if len(labeled_df) == 0:
        fallback = defaults.get(regime_name, DEFAULT_SIMILAR_WEIGHTS)
        print(f"No {regime_name} samples. Using default weights: {fallback}")
        return fallback, 0, 0.0

    print(f"GRID SEARCH: {regime_name.upper()} REGIME ({len(labeled_df)} samples)")

    best_acc = 0
    best_weights = None
    best_correct = 0

    for combo in tqdm(iter_product(*value_lists), total=total, desc=regime_name.capitalize()):
        weights = dict(zip(keys, combo))
        if weights["whisper"] == 0 and weights["seamless"] == 0 and weights["consensus"] == 0:
            continue
        c, t = evaluate_weights(weights, labeled_df, signal_data)
        acc = c / t if t > 0 else 0
        if acc > best_acc or (
            acc == best_acc
            and sum(abs(v) for v in weights.values())
            < sum(abs(v) for v in (best_weights or weights).values())
        ):
            best_acc = acc
            best_weights = weights.copy()
            best_correct = c

    print(f"Best {regime_name.upper()}: {best_correct}/{len(labeled_df)} = {best_acc * 100:.1f}%")
    print(f"Weights: {best_weights}")
    return best_weights, best_correct, best_acc
