"""
Outlier-Aware Consensus Scorer — MBR + outlier detection.
Baseline that should beat vanilla MBR on the real dataset.
"""

from typing import List, Optional
from benchmark.base_scorer import BaseScorer
from golden_asr.scorers.consensus_mbr import ConsensusMBRScorer
from golden_asr.scorers.outlier_detector import OutlierDetector


class OutlierAwareConsensus(BaseScorer):

    @property
    def name(self) -> str:
        return "outlier_aware_consensus"

    def setup(self):
        self.mbr = ConsensusMBRScorer()
        self.outlier = OutlierDetector()

    def score(
        self,
        options: List[str],
        audio_path: Optional[str] = None,
        language: str = "ar",
    ) -> List[float]:
        mask, penalty = self.outlier.detect_with_scores(options)
        raw = self.mbr.score_all(options)
        return [s * p for s, p in zip(raw, penalty)]
