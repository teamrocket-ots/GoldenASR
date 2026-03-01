"""
CSV output generation for the GoldenASR pipeline.
"""

import pandas as pd
from tqdm.auto import tqdm

from golden_asr.preprocessing.filters import is_script_option
from golden_asr.scoring.signals import compute_wer_safe


def build_output_csv(
    df,
    final_predictions,
    signal_data,
    sample_diversity,
    regime_pred,
    output_path,
):
    """Build and save the submission CSV with golden transcription selections.

    Also saves a detailed signal scores CSV alongside the main output.

    Args:
        df: Full dataset DataFrame.
        final_predictions: Dict mapping audio_id to predicted option number.
        signal_data: Full signal data dict.
        sample_diversity: Dict mapping audio_id to diversity score.
        regime_pred: Dict mapping audio_id to regime string.
        output_path: Path for the main output CSV.

    Returns:
        pd.DataFrame: The output DataFrame.
    """
    print("Building output CSV...")
    output_rows = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Output"):
        aid = row["audio_id"]
        golden_opt = int(final_predictions[aid])
        golden_text = str(row[f"option_{golden_opt}"])

        result = {
            "audio_id": aid,
            "language": row["language"],
            "audio": row["audio"],
            "option_1": str(row["option_1"]),
            "option_2": str(row["option_2"]),
            "option_3": str(row["option_3"]),
            "option_4": str(row["option_4"]),
            "option_5": str(row["option_5"]),
            "golden_ref": f"option_{golden_opt}",
            "regime": regime_pred.get(aid, "unknown"),
            "diversity": round(sample_diversity.get(aid, 0.0), 4),
        }

        # WER of each option vs golden
        for j in range(1, 6):
            opt_text = str(row[f"option_{j}"])
            if is_script_option(opt_text) or len(opt_text.strip()) < 2:
                result[f"wer_option{j}"] = 1.0
            elif j == golden_opt:
                result[f"wer_option{j}"] = 0.0
            else:
                result[f"wer_option{j}"] = round(
                    compute_wer_safe(golden_text, opt_text), 4
                )

        # Per-option signal scores
        sigs = signal_data.get(aid, {})
        for j in range(1, 6):
            s = sigs.get(j, {})
            result[f"whisper_sim_option{j}"] = round(s.get("whisper_sim", 0), 4)
            result[f"seamless_sim_option{j}"] = round(s.get("seamless_sim", 0), 4)
            result[f"consensus_option{j}"] = round(s.get("consensus", 0), 4)
            result[f"quality_penalty_option{j}"] = round(
                s.get("quality_penalty", 0), 4
            )
            result[f"hard_filtered_option{j}"] = s.get("hard_filter", True)

        result["source"] = "pipeline_v5"
        output_rows.append(result)

    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"Output saved: {output_path}")
    print(f"Shape: {output_df.shape}")
    print(f"Golden ref distribution:")
    print(output_df["golden_ref"].value_counts().sort_index())

    # Detailed signal scores
    _save_detailed_scores(signal_data, sample_diversity, regime_pred, final_predictions, output_path)

    return output_df


def _save_detailed_scores(
    signal_data, sample_diversity, regime_pred, final_predictions, output_path
):
    """Save a detailed per-option signal scores CSV."""
    detailed_path = output_path.replace(".csv", "_detailed.csv")
    detail_rows = []
    for aid, sigs in signal_data.items():
        for j, s in sigs.items():
            if s.get("is_script", False):
                continue
            detail_rows.append(
                {
                    "audio_id": aid,
                    "option": j,
                    "whisper_sim": round(s.get("whisper_sim", 0), 4),
                    "seamless_sim": round(s.get("seamless_sim", 0), 4),
                    "consensus": round(s.get("consensus", 0), 4),
                    "fluency": round(s.get("fluency", 0), 4),
                    "quality_penalty": round(s.get("quality_penalty", 0), 4),
                    "hard_filter": s.get("hard_filter", True),
                    "regime": regime_pred.get(aid, "unknown"),
                    "diversity": round(sample_diversity.get(aid, 0.0), 4),
                    "is_golden": j == final_predictions.get(aid, -1),
                }
            )
    pd.DataFrame(detail_rows).to_csv(detailed_path, index=False)
    print(f"Detailed scores: {detailed_path}")
