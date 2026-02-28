"""
Whisper Rescoring Scorer — uses Whisper teacher-forced decoding log-probs.
Requires GPU and large-v3 (+ optional small draft model for speculative rescoring).

Usage:
    python run_benchmark.py scorers.whisper_rescore.WhisperRescorer --data data/input.csv
    python run_benchmark.py scorers.whisper_rescore.WhisperRescorer --data data/input.csv --limit 5
"""

from typing import List, Optional

from benchmark.base_scorer import BaseScorer


class WhisperRescorer(BaseScorer):
    """
    Score candidates via Whisper teacher-forced decoding.
    Supports speculative rescoring (small draft filters, large verifies top-K).
    """

    def __init__(
        self,
        model_large: str = "large-v3",
        model_small: str = "small",
        speculative: bool = True,
        language: Optional[str] = None,
    ):
        self._model_large = model_large
        self._model_small = model_small
        self._speculative = speculative
        self._language = language

    @property
    def name(self) -> str:
        spec = "_spec" if self._speculative else ""
        return f"whisper_{self._model_large}{spec}"

    def setup(self):
        from golden_asr.config import Config
        from golden_asr.scorers.whisper_scorer import WhisperEncoder

        cfg = Config(
            whisper_model_large=self._model_large,
            whisper_model_small=self._model_small,
            speculative_enabled=self._speculative,
            whisper_language=self._language,
        )
        self.encoder = WhisperEncoder(cfg)
        self.cfg = cfg

    def teardown(self):
        import torch
        del self.encoder
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

        import torch
        import torchaudio

        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = waveform.mean(dim=0)  # mono

        # Override language per-sample if different from init
        if language and language != self._language:
            import whisper
            self.encoder.tokenizer = whisper.tokenizer.get_tokenizer(
                multilingual=True, language=language, task="transcribe"
            )

        # Encode audio once
        encoder_output = self.encoder.encode_audio(waveform)

        # Speculative or full rescoring
        if self._speculative and self.encoder.model_small is not None:
            scores = self.encoder.speculative_rescore(
                waveform, encoder_output, options
            )
        else:
            scores = [
                self.encoder.score_transcription(encoder_output, text)
                for text in options
            ]

        return scores
