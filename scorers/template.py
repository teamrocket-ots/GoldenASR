"""
TEMPLATE — Copy this file and implement your scorer.

Steps:
  1. cp scorers/template.py scorers/yourname_strategy.py
  2. Rename the class
  3. Implement score()
  4. Run:  python run_benchmark.py scorers.yourname_strategy.YourScorer --data data/input.csv
"""

from typing import List, Optional
from benchmark.base_scorer import BaseScorer


class MyScorer(BaseScorer):

    @property
    def name(self) -> str:
        # Change this to something unique like "jayc_whisper_v3"
        return "my_scorer"

    def setup(self):
        """Load your models here (called once before scoring starts)."""
        # Example:
        # import whisper
        # self.model = whisper.load_model("large-v3")
        pass

    def teardown(self):
        """Free GPU memory etc. (called after scoring finishes)."""
        pass

    def score(
        self,
        options: List[str],
        audio_path: Optional[str] = None,
        language: str = "ar",
    ) -> List[float]:
        """
        YOUR LOGIC HERE.

        Args:
            options:    ['transcription 1', ..., 'transcription 5']
            audio_path: '/path/to/audio.wav' or None
            language:   'ar', 'en', etc.

        Returns:
            [score1, score2, score3, score4, score5]
            Highest score = best transcription.

        You can use audio, text, or both. Some ideas:
            - Whisper forced-decode log-probs
            - CTC alignment confidence
            - LM perplexity
            - Consensus voting
            - Any combination
        """
        # Placeholder: return equal scores
        return [0.5] * len(options)
