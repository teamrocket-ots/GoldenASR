"""
Fusion Ensemble Scorer — combines multiple scoring signals with weighted fusion.
This is the full pipeline scorer that runs all (or selected) sub-scorers
and fuses their outputs.

Usage:
    # Full GPU pipeline (whisper + CTC + MBR + LM)
    python run_benchmark.py scorers.fusion_ensemble.FullPipelineScorer --data data/input.csv

    # Text-only (MBR + LM, no audio needed)
    python run_benchmark.py scorers.fusion_ensemble.TextOnlyFusionScorer --data data/input.csv --no-audio

    # Custom weights
    python run_benchmark.py scorers.fusion_ensemble.FullPipelineScorer --data data/input.csv
"""

import numpy as np
from typing import List, Optional, Dict

from benchmark.base_scorer import BaseScorer


class FusionEnsembleScorer(BaseScorer):
    """
    Configurable multi-signal fusion scorer.
    
    Combines any subset of: whisper rescoring, CTC alignment, MBR consensus, LM fluency.
    Each signal is min-max normalized, optionally inverted, then weighted-summed.
    
    Args:
        weights: dict mapping signal name → weight. E.g. {"mbr": 0.5, "lm": 0.5}
        whisper_model: Whisper model name (None to disable)
        ctc_model: CTC model name (None to disable)
        lm_model: LM model name (None to disable)
        use_mbr: whether to include text consensus
        speculative: whether to use speculative rescoring for Whisper
        language: default language for Whisper tokenizer
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        whisper_model: Optional[str] = "large-v3",
        ctc_model: Optional[str] = "facebook/wav2vec2-large-xlsr-53",
        lm_model: Optional[str] = "aubmindlab/aragpt2-base",
        use_mbr: bool = True,
        speculative: bool = True,
        language: Optional[str] = None,
    ):
        self._weights = weights or {
            "whisper": 0.35,
            "ctc": 0.35,
            "mbr": 0.15,
            "lm": 0.15,
        }
        self._whisper_model = whisper_model
        self._ctc_model = ctc_model
        self._lm_model = lm_model
        self._use_mbr = use_mbr
        self._speculative = speculative
        self._language = language

    @property
    def name(self) -> str:
        parts = []
        if self._whisper_model:
            parts.append("W")
        if self._ctc_model:
            parts.append("C")
        if self._use_mbr:
            parts.append("M")
        if self._lm_model:
            parts.append("L")
        return f"fusion_{''.join(parts)}"

    def setup(self):
        self._scorers = {}

        if self._whisper_model:
            from golden_asr.config import Config
            from golden_asr.scorers.whisper_scorer import WhisperEncoder
            cfg = Config(
                whisper_model_large=self._whisper_model,
                whisper_model_small="small",
                speculative_enabled=self._speculative,
                whisper_language=self._language,
            )
            self._scorers["whisper"] = {"encoder": WhisperEncoder(cfg), "cfg": cfg}

        if self._ctc_model:
            from golden_asr.config import Config
            from golden_asr.scorers.ctc_alignment import CTCAlignmentScorer
            cfg = Config(alignment_model=self._ctc_model)
            self._scorers["ctc"] = CTCAlignmentScorer(cfg)

        if self._use_mbr:
            from golden_asr.scorers.consensus_mbr import ConsensusMBRScorer
            self._scorers["mbr"] = ConsensusMBRScorer()

        if self._lm_model:
            from golden_asr.config import Config
            from golden_asr.scorers.lm_fluency import LMFluencyScorer
            cfg = Config(lm_model=self._lm_model)
            self._scorers["lm"] = LMFluencyScorer(cfg)

    def teardown(self):
        import torch
        self._scorers.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _normalize(scores: List[float], invert: bool = False) -> List[float]:
        """Min-max normalize to [0, 1], optionally invert."""
        arr = np.array(scores, dtype=np.float64)
        if len(arr) == 0:
            return []
        if np.all(arr == arr[0]):
            return [0.5] * len(arr)
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return [0.5] * len(arr)
        normed = (arr - mn) / (mx - mn)
        if invert:
            normed = 1.0 - normed
        return normed.tolist()

    def score(
        self,
        options: List[str],
        audio_path: Optional[str] = None,
        language: str = "ar",
    ) -> List[float]:
        n = len(options)
        raw_signals: Dict[str, List[float]] = {}

        # 1) MBR (always runs first — cheapest)
        if "mbr" in self._scorers:
            raw_signals["mbr"] = self._scorers["mbr"].score_all(options)

        # 2) LM fluency (text-only, no audio)
        if "lm" in self._scorers:
            raw_signals["lm"] = self._scorers["lm"].score_all(options)

        # Audio-dependent scorers
        waveform = None
        if audio_path:
            import torchaudio
            waveform_raw, sr = torchaudio.load(audio_path)
            if sr != 16000:
                waveform_raw = torchaudio.functional.resample(waveform_raw, sr, 16000)
            waveform = waveform_raw.mean(dim=0)  # mono

        # 3) Whisper rescoring
        if "whisper" in self._scorers and waveform is not None:
            enc_data = self._scorers["whisper"]
            encoder = enc_data["encoder"]
            cfg = enc_data["cfg"]

            # Update language if needed
            if language and language != self._language:
                import whisper
                encoder.tokenizer = whisper.tokenizer.get_tokenizer(
                    multilingual=True, language=language, task="transcribe"
                )

            encoder_output = encoder.encode_audio(waveform)

            if self._speculative and encoder.model_small is not None:
                raw_signals["whisper"] = encoder.speculative_rescore(
                    waveform, encoder_output, options
                )
            else:
                raw_signals["whisper"] = [
                    encoder.score_transcription(encoder_output, text)
                    for text in options
                ]

        # 4) CTC alignment
        if "ctc" in self._scorers and waveform is not None:
            raw_signals["ctc"] = self._scorers["ctc"].score_all(waveform, options)

        # Fuse: normalize and weighted sum
        if not raw_signals:
            return [0.5] * n

        final = np.zeros(n)
        total_w = 0.0

        for sig_name, scores in raw_signals.items():
            w = self._weights.get(sig_name, 0.0)
            if w <= 0:
                continue
            # LM: lower NLL = better → invert
            invert = sig_name == "lm"
            normed = self._normalize(scores, invert=invert)
            final += w * np.array(normed)
            total_w += w

        if total_w > 0:
            final /= total_w

        return final.tolist()


# ── Pre-configured variants for easy CLI use ─────────────────────────


class FullPipelineScorer(FusionEnsembleScorer):
    """Full GPU pipeline: Whisper + CTC + MBR + LM."""

    def __init__(self):
        super().__init__(
            weights={"whisper": 0.35, "ctc": 0.35, "mbr": 0.15, "lm": 0.15},
            whisper_model="large-v3",
            ctc_model="facebook/wav2vec2-large-xlsr-53",
            lm_model="aubmindlab/aragpt2-base",
            use_mbr=True,
            speculative=True,
        )

    @property
    def name(self) -> str:
        return "full_pipeline"


class TextOnlyFusionScorer(FusionEnsembleScorer):
    """Text-only: MBR + LM (no audio needed)."""

    def __init__(self):
        super().__init__(
            weights={"mbr": 0.6, "lm": 0.4},
            whisper_model=None,
            ctc_model=None,
            lm_model="aubmindlab/aragpt2-base",
            use_mbr=True,
        )

    @property
    def name(self) -> str:
        return "text_only_mbr_lm"


class AcousticOnlyScorer(FusionEnsembleScorer):
    """Acoustic-only: Whisper + CTC (no text signals)."""

    def __init__(self):
        super().__init__(
            weights={"whisper": 0.5, "ctc": 0.5},
            whisper_model="large-v3",
            ctc_model="facebook/wav2vec2-large-xlsr-53",
            lm_model=None,
            use_mbr=False,
            speculative=True,
        )

    @property
    def name(self) -> str:
        return "acoustic_only"
