#!/usr/bin/env python3
"""
Run a benchmark against a scorer.

Usage:
    python run_benchmark.py scorers.consensus_baseline.ConsensusBaseline
    python run_benchmark.py scorers.consensus_baseline.ConsensusBaseline --data data/input.csv
    python run_benchmark.py scorers.consensus_baseline.ConsensusBaseline --limit 10
    python run_benchmark.py scorers.random_baseline.RandomScorer --no-audio
"""

import argparse
import importlib
import sys

sys.path.insert(0, ".")


def main():
    parser = argparse.ArgumentParser(
        description="GoldenASR Benchmark — test your scorer against the dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py scorers.consensus_baseline.ConsensusBaseline
  python run_benchmark.py scorers.my_scorer.MyScorer --data data/input.csv --limit 20
  python run_benchmark.py scorers.random_baseline.RandomScorer --no-audio

Your scorer must subclass benchmark.base_scorer.BaseScorer.
See scorers/template.py for a starting point.
""",
    )
    parser.add_argument(
        "scorer",
        type=str,
        help="Dotted path to scorer class (e.g., scorers.my_scorer.MyScorer)",
    )
    parser.add_argument(
        "--data", "-d",
        default="data/input.csv",
        help="Path to input CSV (default: data/input.csv)",
    )
    parser.add_argument(
        "--audio-dir",
        default="data/audio",
        help="Audio cache directory (default: data/audio)",
    )
    parser.add_argument(
        "--output", "-o",
        default="benchmark_results",
        help="Output directory (default: benchmark_results)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Only process first N samples (for quick testing)",
    )
    parser.add_argument(
        "--no-audio",
        action="store_true",
        help="Skip audio download (for text-only scorers)",
    )
    parser.add_argument(
        "--reference", "-r",
        default=None,
        help="CSV with ground-truth labels (columns: audio_id, reference). "
             "Enables real WER/accuracy evaluation. If your data CSV already has "
             "a 'reference' column, this is detected automatically.",
    )
    args = parser.parse_args()

    # Dynamically import the scorer class
    parts = args.scorer.rsplit(".", 1)
    if len(parts) != 2:
        print(f"Error: scorer must be 'module.ClassName', got '{args.scorer}'")
        sys.exit(1)

    module_path, class_name = parts
    try:
        module = importlib.import_module(module_path)
        scorer_cls = getattr(module, class_name)
    except (ImportError, AttributeError) as e:
        print(f"Error loading scorer '{args.scorer}': {e}")
        sys.exit(1)

    scorer = scorer_cls()

    from benchmark.runner import run_benchmark

    run_benchmark(
        scorer=scorer,
        data_csv=args.data,
        audio_dir=args.audio_dir,
        output_dir=args.output,
        download_audio=not args.no_audio,
        limit=args.limit,
        reference_csv=args.reference,
    )


if __name__ == "__main__":
    main()
