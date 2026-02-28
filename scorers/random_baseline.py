"""
Random Scorer — for sanity checking. Picks randomly.
Your scorer must beat this or something is very wrong.
"""

import random
from typing import List, Optional
from benchmark.base_scorer import BaseScorer


class RandomScorer(BaseScorer):

    @property
    def name(self) -> str:
        return "random_baseline"

    def score(
        self,
        options: List[str],
        audio_path: Optional[str] = None,
        language: str = "ar",
    ) -> List[float]:
        return [random.random() for _ in options]
