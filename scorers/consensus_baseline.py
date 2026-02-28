"""
Consensus MBR Baseline — the simplest scorer.
Uses only text similarity between candidates (no audio, no models).
This is the baseline every other scorer should beat.
"""

from typing import List, Optional
from benchmark.base_scorer import BaseScorer
from golden_asr.scorers.consensus_mbr import ConsensusMBRScorer


class ConsensusBaseline(BaseScorer):

    @property
    def name(self) -> str:
        return "consensus_baseline"

    def setup(self):
        self.mbr = ConsensusMBRScorer()

    def score(
        self,
        options: List[str],
        audio_path: Optional[str] = None,
        language: str = "ar",
    ) -> List[float]:
        return self.mbr.score_all(options)
