"""
Whisper Pseudo-Reference Scorer — our strongest evaluation signal without GT.

Strategy:
  1. Transcribe each audio with Whisper large-v3 to get a "pseudo-reference"
  2. Score each of the 5 options by CER against this pseudo-reference
  3. The option closest to Whisper's own transcription scores highest

This is the primary no-GT evaluation method. While Whisper's own output isn't
ground truth, it's a strong enough signal to compare scorer strategies.

Requires: GPU + openai-whisper

Usage:
    python run_benchmark.py scorers.whisper_pseudo_ref.WhisperPseudoRefScorer \
        --data data/input.csv --audio-dir data/audio
"""

from typing import List, Optional
from benchmark.base_scorer import BaseScorer


class WhisperPseudoRefScorer(BaseScorer):
    """
    Score options by similarity to Whisper's own transcription.
    
    This is NOT cheating — Whisper produces a 6th independent transcription,
    and we measure which of the 5 options is closest to it. The assumption
    is that a good ASR model's transcription correlates with the true reference.
    """

    def __init__(self, model_size: str = "large-v3", language: str = "ar"):
        self.model_size = model_size
        self._language = language

    @property
    def name(self) -> str:
        return "whisper_pseudo_ref_{}".format(self.model_size.replace("-", ""))

    def setup(self):
        import whisper
        print("[WhisperPseudoRef] Loading Whisper {} ...".format(self.model_size))
        self.model = whisper.load_model(self.model_size)
        print("[WhisperPseudoRef] Model loaded.")

    def teardown(self):
        if hasattr(self, "model"):
            del self.model

    def _transcribe(self, audio_path: str) -> str:
        """Transcribe audio file using Whisper."""
        result = self.model.transcribe(
            audio_path,
            language=self._language,
            task="transcribe",
            without_timestamps=True,
        )
        return result["text"].strip()

    def score(
        self,
        options: List[str],
        audio_path: Optional[str] = None,
        language: str = "ar",
    ) -> List[float]:
        from golden_asr.scorers.outlier_detector import OutlierDetector
        from rapidfuzz.distance import Levenshtein

        # Pre-filter outliers
        detector = OutlierDetector()
        mask, penalty = detector.detect_with_scores(options)

        if audio_path is None:
            # Fallback: without audio, just use outlier + MBR
            from golden_asr.scorers.consensus_mbr import ConsensusMBRScorer
            mbr = ConsensusMBRScorer()
            scores = mbr.score_all(options)
            return [s * p for s, p in zip(scores, penalty)]

        # Transcribe with Whisper
        pseudo_ref = self._transcribe(audio_path)

        # Score each option by similarity to pseudo-reference
        scores = []
        for i, opt in enumerate(options):
            if not mask[i]:
                scores.append(0.0)  # Outlier gets 0
            else:
                similarity = Levenshtein.normalized_similarity(pseudo_ref, opt)
                scores.append(similarity)

        return scores


class WhisperPseudoRefSmall(WhisperPseudoRefScorer):
    """Smaller model for faster evaluation."""

    def __init__(self):
        super().__init__(model_size="small", language="ar")


class WhisperPseudoRefMedium(WhisperPseudoRefScorer):
    """Medium model — balance of speed and quality."""

    def __init__(self):
        super().__init__(model_size="medium", language="ar")


class WhisperPseudoRefLarge(WhisperPseudoRefScorer):
    """Large-v3 model — best quality but slow."""

    def __init__(self):
        super().__init__(model_size="large-v3", language="ar")
