"""
Signal computation for candidate transcription scoring.

Each candidate option receives a vector of scoring signals that are later
combined via learned weights to select the golden transcription.
"""

import numpy as np
from jiwer import wer
from tqdm.auto import tqdm

from golden_asr.preprocessing.filters import (
    compute_quality_penalty,
    has_scene_markers,
    has_speaker_labels,
    has_stage_directions,
    is_script_option,
)
from golden_asr.preprocessing.normalization import normalize_text


def compute_wer_safe(ref, hyp):
    """Compute WER between two texts with safe normalization.

    Both inputs are normalized before comparison. Returns 1.0 if either
    input is empty after normalization.

    Args:
        ref: Reference text.
        hyp: Hypothesis text.

    Returns:
        float: Word Error Rate clipped to [0.0, 1.0].
    """
    r = normalize_text(ref)
    h = normalize_text(hyp)
    if not r or not h:
        return 1.0
    if r == h:
        return 0.0
    try:
        return min(wer(r, h), 1.0)
    except Exception:
        return 1.0


def compute_similarity(ref, hyp):
    """Compute text similarity as 1 - WER.

    Args:
        ref: Reference text.
        hyp: Hypothesis text.

    Returns:
        float: Similarity in [0.0, 1.0].
    """
    return 1.0 - compute_wer_safe(ref, hyp)


def compute_signals_for_sample(options, whisper_ref, seamless_ref):
    """Compute all scoring signals for one sample's candidate options.

    Args:
        options: Dict mapping option number (1-5) to option text.
        whisper_ref: Whisper ASR transcription for this sample.
        seamless_ref: SeamlessM4T transcription for this sample.

    Returns:
        dict: Mapping of option number to signal dict.
    """
    valid = {
        j: t
        for j, t in options.items()
        if not is_script_option(t) and len(t.strip()) >= 5
    }

    row_signals = {}
    for j, text in options.items():
        sig = {}

        # Mark scripts/empty as filtered
        if is_script_option(text) or len(text.strip()) < 5:
            sig.update(
                {
                    "is_script": True,
                    "hard_filter": True,
                    "whisper_sim": 0.0,
                    "seamless_sim": 0.0,
                    "consensus": 0.0,
                    "fluency": 0.0,
                    "quality_penalty": 1.0,
                }
            )
            row_signals[j] = sig
            continue

        sig["is_script"] = False
        sig["has_speaker"] = has_speaker_labels(text)
        sig["has_stage"] = has_stage_directions(text)
        sig["has_scene"] = has_scene_markers(text)
        sig["hard_filter"] = sig["has_speaker"] or sig["has_stage"] or sig["has_scene"]

        # Signal 1: Whisper similarity
        sig["whisper_sim"] = compute_similarity(whisper_ref, text) if whisper_ref else 0.5

        # Signal 2: SeamlessM4T similarity
        sig["seamless_sim"] = (
            compute_similarity(seamless_ref, text) if seamless_ref else 0.5
        )

        # Signal 3: Consensus (avg pairwise similarity with other valid options)
        pairwise = [compute_similarity(text, other) for k, other in valid.items() if k != j]
        sig["consensus"] = np.mean(pairwise) if pairwise else 0.5

        # Signal 4: Fluency proxy
        words = normalize_text(text).split()
        sig["fluency"] = (
            len(words) / max(len(normalize_text(text)), 1) if words else 0.0
        )

        # Signal 5: Quality penalty
        sig["quality_penalty"] = compute_quality_penalty(text)

        # Signal 6: Text length (for relative length normalization)
        sig["text_length"] = len(text)

        row_signals[j] = sig

    # Normalize relative length across valid options
    valid_lengths = [row_signals[j].get("text_length", 0) for j in valid]
    if valid_lengths:
        max_len = max(valid_lengths) if max(valid_lengths) > 0 else 1
        for j in valid:
            row_signals[j]["rel_length"] = (
                row_signals[j].get("text_length", 0) / max_len
            )

    return row_signals


def compute_all_signals(df, whisper_texts, seamless_texts):
    """Compute scoring signals for every sample in the dataset.

    Args:
        df: Dataset DataFrame with ``audio_id`` and ``option_1``..``option_5``.
        whisper_texts: Dict mapping audio_id to Whisper transcription.
        seamless_texts: Dict mapping audio_id to SeamlessM4T transcription.

    Returns:
        dict: Mapping of audio_id to (dict of option_num to signal dict).
    """
    print(f"Computing scoring signals for {len(df)} samples x 5 options...")
    signal_data = {}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Scoring"):
        aid = row["audio_id"]
        w_ref = whisper_texts.get(aid, "")
        s_ref = seamless_texts.get(aid, "")
        options = {j: str(row[f"option_{j}"]) for j in range(1, 6)}
        signal_data[aid] = compute_signals_for_sample(options, w_ref, s_ref)

    # Summary stats
    pass_counts = [
        sum(1 for s in sigs.values() if not s.get("hard_filter", True))
        for sigs in signal_data.values()
    ]
    print(f"Scoring complete for {len(signal_data)} samples.")
    print(
        f"Options passing hard filter: mean={np.mean(pass_counts):.1f}, "
        f"min={min(pass_counts)}, max={max(pass_counts)}"
    )

    return signal_data
