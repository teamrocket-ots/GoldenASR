"""
BaseScorer — the interface every team member must implement.

Your scorer receives the 5 transcription options (and optionally audio)
and returns a list of 5 scores. Highest score wins = golden.

That's it. Implement `score()`, drop your file in scorers/, run benchmark.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseScorer(ABC):
    """
    Implement this class to create your scoring strategy.

    Required:
        name        — a short identifier (e.g., "jayc_whisper_v2")
        score()     — given options (and optional audio), return scores

    Optional:
        setup()     — load models once before the benchmark loop
        teardown()  — cleanup after benchmark finishes
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short unique name for this scorer (used in output filenames)."""
        ...

    def setup(self):
        """Called once before scoring starts. Load models here."""
        pass

    def teardown(self):
        """Called after all scoring is done. Free resources here."""
        pass

    @abstractmethod
    def score(
        self,
        options: List[str],
        audio_path: Optional[str] = None,
        language: str = "ar",
    ) -> List[float]:
        """
        Score the transcription candidates. Return one float per option.
        HIGHER score = BETTER transcription.

        Args:
            options:    list of 5 transcription strings
            audio_path: path to the .wav file (None if audio unavailable)
            language:   ISO language code (e.g., "ar", "en")

        Returns:
            List of 5 floats — one score per option. Highest wins.
        """
        ...
