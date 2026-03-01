"""
Command-line entry point for the GoldenASR pipeline.

Usage:
    python -m golden_asr --csv path/to/dataset.csv --output output.csv
"""

import argparse

from golden_asr.pipeline import run


def main():
    parser = argparse.ArgumentParser(
        description="GoldenASR: Golden Transcription Selection Pipeline v5"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to the dataset CSV. Auto-detected on Kaggle if omitted.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the output submission CSV.",
    )
    parser.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Directory for downloaded audio files.",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Path for the analysis panel PNG.",
    )
    args = parser.parse_args()

    run(
        csv_path=args.csv,
        output_csv=args.output,
        audio_dir=args.audio_dir,
        plot_path=args.plot,
    )


if __name__ == "__main__":
    main()
