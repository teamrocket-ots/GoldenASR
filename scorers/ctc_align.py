"""
CTC Alignment Scorer — forced alignment confidence via wav2vec2-xlsr-53.
Requires GPU and transformers.

Usage:
    python run_benchmark.py scorers.ctc_align.CTCAlignScorer --data data/input.csv
"""

from typing import List, Optional

from benchmark.base_scorer import BaseScorer


class CTCAlignScorer(BaseScorer):
    """
    Score candidates by CTC forced-alignment acoustic confidence.
    Uses wav2vec2-large-xlsr-53 (multilingual CTC model).
    """

    def __init__(self, model: str = "facebook/wav2vec2-large-xlsr-53"):
        self._model = model

    @property
    def name(self) -> str:
        short = self._model.split("/")[-1]
        return f"ctc_{short}"

    def setup(self):
        from golden_asr.config import Config
        from golden_asr.scorers.ctc_alignment import CTCAlignmentScorer

        cfg = Config(alignment_model=self._model)
        self.scorer = CTCAlignmentScorer(cfg)

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
        if audio_path is None:
            return [0.0] * len(options)

        import torchaudio

        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        waveform = waveform.mean(dim=0)  # mono

        return self.scorer.score_all(waveform, options)
