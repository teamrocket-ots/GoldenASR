"""
Dataset loading utilities for the GoldenASR pipeline.
"""

import glob
import numpy as np
import pandas as pd


def parse_correct_option(val):
    """Parse a correct_option value to an integer.

    Handles formats such as ``option_1``, ``"1"``, and ``1.0``.

    Args:
        val: Raw correct_option value from the CSV.

    Returns:
        int or np.nan: Parsed integer option number, or NaN if unparseable.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s.startswith("option_"):
        return int(s.replace("option_", ""))
    try:
        return int(float(s))
    except ValueError:
        return np.nan


def auto_detect_csv(search_patterns=None):
    """Attempt to auto-detect the dataset CSV path on Kaggle.

    Args:
        search_patterns: Optional list of glob patterns to search.
            Defaults to common Kaggle input patterns.

    Returns:
        str or None: First matching CSV path, or None if nothing found.
    """
    if search_patterns is None:
        search_patterns = [
            "/kaggle/input/*/Transcription Assessment Arabic_SA Dataset*.csv",
            "/kaggle/input/*/transcription*arabic*.csv",
            "/kaggle/input/*/*.csv",
        ]
    for pattern in search_patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[0]
    return None


def load_dataset(csv_path):
    """Load the transcription assessment dataset from a CSV file.

    The returned DataFrame will include a ``correct_option_int`` column
    containing the integer-parsed correct option (NaN for unlabeled rows).

    Args:
        csv_path: Path to the dataset CSV.

    Returns:
        pd.DataFrame: Loaded and augmented dataset.
    """
    df = pd.read_csv(csv_path)
    df["correct_option_int"] = df["correct_option"].apply(parse_correct_option)
    return df
