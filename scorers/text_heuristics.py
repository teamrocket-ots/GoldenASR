"""
Text Heuristics Scorer — combine multiple text-only signals.
No GPU or models needed. Good for fast iteration.

Signals:
  1. MBR consensus (pairwise CER)
  2. Length normality (prefer median-length transcription)
  3. Character diversity (penalize repetitive text)
  4. Word count consistency

Usage:
    python run_benchmark.py scorers.text_heuristics.TextHeuristicsScorer --data data/input.csv --no-audio
"""

import math
import statistics
from typing import List, Optional
from collections import Counter

from benchmark.base_scorer import BaseScorer
from golden_asr.scorers.consensus_mbr import ConsensusMBRScorer


class TextHeuristicsScorer(BaseScorer):
    """
    Weighted combination of cheap text-only heuristics.
    Useful as a strong no-GPU baseline.
    """

    def __init__(
        self,
        w_mbr: float = 0.50,
        w_length: float = 0.20,
        w_diversity: float = 0.15,
        w_wordcount: float = 0.15,
    ):
        self.w_mbr = w_mbr
        self.w_length = w_length
        self.w_diversity = w_diversity
        self.w_wordcount = w_wordcount

    @property
    def name(self) -> str:
        return "text_heuristics"

    def setup(self):
        self.mbr = ConsensusMBRScorer()

    def _length_score(self, options: List[str]) -> List[float]:
        """
        Score by proximity to median length.
        Idea: outlier lengths are likely errors.
        """
        lengths = [len(opt) for opt in options]
        if not lengths or max(lengths) == 0:
            return [0.5] * len(options)
        med = statistics.median(lengths)
        if med == 0:
            return [0.5] * len(options)
        # Score = 1 - |len - median| / median, clamped to [0, 1]
        scores = []
        for l in lengths:
            deviation = abs(l - med) / med
            scores.append(max(0.0, 1.0 - deviation))
        return scores

    def _diversity_score(self, options: List[str]) -> List[float]:
        """
        Score by character diversity (unique chars / total chars).
        Very repetitive text (e.g., "aaaaaaa") gets low score.
        """
        scores = []
        for opt in options:
            if len(opt) < 2:
                scores.append(0.5)
                continue
            unique = len(set(opt))
            ratio = unique / len(opt)
            # Normalize: typical ratio ~0.3-0.7 for natural text
            # Score peaks at ~0.5 diversity ratio
            scores.append(min(1.0, ratio * 2))
        return scores

    def _word_count_score(self, options: List[str]) -> List[float]:
        """
        Score by proximity to median word count.
        """
        word_counts = [len(opt.split()) for opt in options]
        if not word_counts or max(word_counts) == 0:
            return [0.5] * len(options)
        med = statistics.median(word_counts)
        if med == 0:
            return [0.5] * len(options)
        scores = []
        for wc in word_counts:
            deviation = abs(wc - med) / med
            scores.append(max(0.0, 1.0 - deviation))
        return scores

    def score(
        self,
        options: List[str],
        audio_path: Optional[str] = None,
        language: str = "ar",
    ) -> List[float]:
        n = len(options)

        mbr_scores = self.mbr.score_all(options)
        length_scores = self._length_score(options)
        diversity_scores = self._diversity_score(options)
        wordcount_scores = self._word_count_score(options)

        final = []
        for i in range(n):
            s = (
                self.w_mbr * mbr_scores[i]
                + self.w_length * length_scores[i]
                + self.w_diversity * diversity_scores[i]
                + self.w_wordcount * wordcount_scores[i]
            )
            final.append(s)

        return final
