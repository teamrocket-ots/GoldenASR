"""
Whisper Reference Scorer — generate a Whisper transcription, then pick
the candidate closest to it by edit distance. Simple but effective.

Does NOT require teacher-forcing — just generates + compares.

Usage:
    python run_benchmark.py scorers.whisper_reference.WhisperRefScorer --data data/input.csv
    python run_benchmark.py scorers.whisper_reference.WhisperRefSmall --data data/input.csv
"""

from typing import List, Optional

from benchmark.base_scorer import BaseScorer


class WhisperRefScorer(BaseScorer):
    """
    1. Generate a reference transcription with Whisper
    2. Score each candidate by (1 - CER) against the reference
    Closest to Whisper's own output wins.
    """

    def __init__(self, model: str = "large-v3", language: Optional[str] = None):
        self._model = model
        self._language = language

    @property
    def name(self) -> str:
        return f"whisper_ref_{self._model}"

    def setup(self):
        import whisper
        device = "cuda"
        try:
            import torch
            if not torch.cuda.is_available():
                device = "cpu"
        except ImportError:
            device = "cpu"

        print(f"[WhisperRef] Loading {self._model} on {device}...")
        self.model = whisper.load_model(self._model, device=device)
        self.model.eval()
        self.device = device

    def teardown(self):
        import torch
        del self.model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def score(
        self,
        options: List[str],
        audio_path: Optional[str] = None,
        language: str = "ar",
    ) -> List[float]:
        if audio_path is None:
            return [0.0] * len(options)

        import whisper
        from rapidfuzz.distance import Levenshtein

        # Generate reference transcription
        result = self.model.transcribe(
            audio_path,
            language=language or self._language,
            fp16=(self.device == "cuda"),
        )
        reference = result["text"].strip()

        if not reference:
            return [0.5] * len(options)

        # Score each candidate by character similarity to reference
        scores = []
        for opt in options:
            if not opt.strip():
                scores.append(0.0)
            else:
                # 1 - CER = similarity
                sim = 1.0 - Levenshtein.normalized_distance(reference, opt)
                scores.append(sim)

        return scores


class WhisperRefSmall(WhisperRefScorer):
    """Same strategy but with the smaller/faster model."""

    def __init__(self):
        super().__init__(model="small")

    @property
    def name(self) -> str:
        return "whisper_ref_small"


class WhisperRefMedium(WhisperRefScorer):
    """Medium model — balance speed & quality."""

    def __init__(self):
        super().__init__(model="medium")

    @property
    def name(self) -> str:
        return "whisper_ref_medium"
