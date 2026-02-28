"""
LM Fluency Scorer — AraGPT2 perplexity with EAGLE-style prefix KV-cache reuse.
Requires GPU and transformers.

Usage:
    python run_benchmark.py scorers.lm_perplexity.LMPerplexityScorer --data data/input.csv --no-audio
"""

from typing import List, Optional

from benchmark.base_scorer import BaseScorer


class LMPerplexityScorer(BaseScorer):
    """
    Score candidates by language model fluency (lower perplexity = better).
    Uses EAGLE-inspired shared-prefix KV-cache optimization.
    
    Note: LM scorer returns NLL (lower=better) internally, but this wrapper
    INVERTS the scores so higher=better for the benchmark interface.
    """

    def __init__(self, model: str = "aubmindlab/aragpt2-base"):
        self._model = model

    @property
    def name(self) -> str:
        short = self._model.split("/")[-1]
        return f"lm_{short}"

    def setup(self):
        from golden_asr.config import Config
        from golden_asr.scorers.lm_fluency import LMFluencyScorer

        cfg = Config(lm_model=self._model)
        self.scorer = LMFluencyScorer(cfg)

    def teardown(self):
        import torch
        del self.scorer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def score(
        self,
        options: List[str],
        audio_path: Optional[str] = None,
        language: str = "ar",
    ) -> List[float]:
        # LM scoring is text-only — audio not needed
        nll_scores = self.scorer.score_all(options)

        # Invert: lower NLL = better fluency → higher output score
        # Use negation so most-fluent gets highest score
        return [-s for s in nll_scores]
