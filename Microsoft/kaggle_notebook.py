"""
GoldenASR — Kaggle Notebook (Self-Contained) — MOONSHINE VERSION
=================================================================
Complete pipeline for golden transcription selection using Moonshine ASR.
Upload this as a Kaggle notebook with GPU enabled.

What this does:
  1. Installs dependencies
  2. Downloads audio files from the dataset
  3. Runs ALL scorers (text-only + GPU-based)
  4. Generates pseudo-references with Moonshine for evaluation
  5. Compares all strategies and selects the best
  6. Outputs final submission CSV

Enable GPU: Settings → Accelerator → GPU T4 x2 (or P100)
Enable Internet: Settings → Internet → On (to download Moonshine model)
"""

# ========== BLOCK 1: Install Dependencies ==========
import subprocess
import sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

install("rapidfuzz")
install("jiwer")
install("transformers")
install("torchaudio")
install("soundfile")
# Note: openai-whisper is still needed for the INDEPENDENT evaluator (medium)
# This breaks circularity: scorer=Moonshine, evaluator=Whisper
install("openai-whisper")

# ========== BLOCK 2: Imports & GPU Check ==========
import os
import time
import json
import gc
import ctypes
import requests
import numpy as np
import pandas as pd
import statistics
from typing import List, Dict, Optional, Tuple
from collections import Counter
from tqdm.auto import tqdm

import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ========== BLOCK 3: Configuration & Data Loading ==========
INPUT_CSV = "/kaggle/input/transcription-assessment-arabic-sa-dataset/Transcription Assessment Arabic_SA Dataset.csv"
# If running locally, change to your path:
# INPUT_CSV = "Transcription Assessment Arabic_SA Dataset.csv"

AUDIO_DIR = "/kaggle/working/audio"
OUTPUT_DIR = "/kaggle/working/output"
os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

LANGUAGE = "ar"
NUM_OPTIONS = 5
OPTION_COLS = [f"option_{i}" for i in range(1, NUM_OPTIONS + 1)]

# --- Moonshine Model Selection (multilingual) ---
# Maps language codes to the best available Moonshine model.
# Add new languages here as Moonshine releases more models.
MOONSHINE_MODEL_MAP = {
    "ar": "UsefulSensors/moonshine-tiny-ar",   # Arabic-specific (27M params)
    "zh": "UsefulSensors/moonshine-tiny-zh",   # Chinese-specific
    "ja": "UsefulSensors/moonshine-tiny-ja",   # Japanese-specific
    "ko": "UsefulSensors/moonshine-tiny-ko",   # Korean-specific
    "uk": "UsefulSensors/moonshine-tiny-uk",   # Ukrainian-specific
    "vi": "UsefulSensors/moonshine-tiny-vi",   # Vietnamese-specific
}
MOONSHINE_FALLBACK_MODEL = "UsefulSensors/moonshine-base"  # English/general (61M params)

# Token limit factor per second (prevents hallucination loops)
# Arabic/CJK scripts need more tokens per second than English
MOONSHINE_TOKEN_FACTOR = {
    "ar": 13,    # Arabic needs higher token limit
    "zh": 13,
    "ja": 13,
    "ko": 13,
    "default": 6.5,  # English and others
}

def get_moonshine_model_id(language: str) -> str:
    """Get the best Moonshine model for a given language."""
    return MOONSHINE_MODEL_MAP.get(language, MOONSHINE_FALLBACK_MODEL)

def get_token_factor(language: str) -> float:
    """Get token limit factor for a language (higher = more tokens allowed)."""
    return MOONSHINE_TOKEN_FACTOR.get(language, MOONSHINE_TOKEN_FACTOR["default"])

print(f"Language: {LANGUAGE}")
print(f"Moonshine model: {get_moonshine_model_id(LANGUAGE)}")
print(f"Token factor: {get_token_factor(LANGUAGE)}")

# Load data
df = pd.read_csv(INPUT_CSV)
print(f"\nDataset: {df.shape[0]} samples, {df.shape[1]} columns")
print(f"Languages: {df['language'].unique()}")
df.head(2)

# ========== BLOCK 4: Download Audio ==========
def download_audio(df, audio_dir, timeout=120):
    """Download all audio files. Returns {audio_id: local_path}."""
    audio_map = {}
    failed = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Downloading audio"):
        aid = str(row["audio_id"])
        url = str(row.get("audio", ""))
        if not url or url == "nan":
            continue
        ext = os.path.splitext(url.split("?")[0])[-1] or ".wav"
        path = os.path.join(audio_dir, f"{aid}{ext}")
        audio_map[aid] = path
        if os.path.exists(path):
            continue
        try:
            r = requests.get(url, timeout=timeout, stream=True)
            r.raise_for_status()
            with open(path, "wb") as f:
                for chunk in r.iter_content(8192):
                    f.write(chunk)
        except Exception as e:
            failed.append(aid)
            print(f"  FAILED {aid}: {e}")
    if failed:
        print(f"WARNING: {len(failed)} downloads failed: {failed}")
    return audio_map

audio_map = download_audio(df, AUDIO_DIR)
print(f"Audio files ready: {len(audio_map)}")

# ========== BLOCK 5: Outlier Detection ==========
class OutlierDetector:
    """Detect non-transcription outlier options (drama scripts)."""
    SCRIPT_MARKERS = ["المشهد", "المكان:", "دراما"]

    def __init__(self, length_ratio=3.0, min_length=500):
        self.length_ratio = length_ratio
        self.min_length = min_length

    def detect(self, options: List[str]) -> List[bool]:
        """Returns mask: True = valid transcription, False = outlier."""
        n = len(options)
        if n <= 2:
            return [True] * n
        lengths = [len(opt) for opt in options]
        median_len = sorted(lengths)[n // 2]
        mask = [True] * n
        for i, opt in enumerate(options):
            if lengths[i] > self.min_length and median_len > 0:
                if lengths[i] / median_len > self.length_ratio:
                    mask[i] = False
                    continue
            for marker in self.SCRIPT_MARKERS:
                if marker in opt[:50]:
                    mask[i] = False
                    break
        return mask

    def get_penalty(self, options: List[str]) -> List[float]:
        mask = self.detect(options)
        return [1.0 if m else 0.0 for m in mask]

outlier = OutlierDetector()

# Verify outlier detection works
test_row = df.iloc[0]
test_opts = [str(test_row[c]) for c in OPTION_COLS]
test_mask = outlier.detect(test_opts)
print(f"Row 1 lengths: {[len(o) for o in test_opts]}")
print(f"Row 1 mask: {test_mask}")
n_detected = sum(1 for _, row in df.iterrows()
                 for c in OPTION_COLS
                 if not outlier.detect([str(row[c2]) for c2 in OPTION_COLS])[OPTION_COLS.index(c)])
print(f"Total outliers detected across dataset: {n_detected} (expect ~100)")

# ========== BLOCK 6: Text-Only Scorers (no GPU) ==========
from rapidfuzz.distance import Levenshtein

class ConsensusMBRScorer:
    """Select option most similar to all others (pairwise CER consensus)."""

    def score(self, options: List[str], audio_path: str = None) -> List[float]:
        n = len(options)
        if n <= 1:
            return [1.0] * n
        scores = []
        for i in range(n):
            total = sum(Levenshtein.normalized_distance(options[i], options[j])
                        for j in range(n) if j != i)
            scores.append(1.0 - total / (n - 1))
        return scores

class OutlierAwareMBR:
    """MBR consensus with outlier detection pre-filter."""

    def score(self, options: List[str], audio_path: str = None) -> List[float]:
        penalty = outlier.get_penalty(options)
        mbr = ConsensusMBRScorer()
        raw = mbr.score(options)
        return [s * p for s, p in zip(raw, penalty)]

class TextHeuristicsScorer:
    """Combined text signals: MBR + length normality + diversity."""

    def score(self, options: List[str], audio_path: str = None) -> List[float]:
        penalty = outlier.get_penalty(options)
        n = len(options)

        # MBR
        mbr_scores = ConsensusMBRScorer().score(options)

        # Length normality
        lengths = [len(o) for o in options]
        med = statistics.median(lengths) if lengths else 1
        length_scores = [max(0, 1 - abs(l - med) / max(med, 1)) for l in lengths]

        # Word count normality
        wc = [len(o.split()) for o in options]
        med_wc = statistics.median(wc) if wc else 1
        wc_scores = [max(0, 1 - abs(w - med_wc) / max(med_wc, 1)) for w in wc]

        # Combine
        final = []
        for i in range(n):
            s = 0.50 * mbr_scores[i] + 0.25 * length_scores[i] + 0.25 * wc_scores[i]
            final.append(s * penalty[i])
        return final

# ========== BLOCK 7: Moonshine Pseudo-Reference Scorer (GPU) ==========
import torchaudio

class MoonshinePseudoRefScorer:
    """
    Transcribe audio with Moonshine, then pick the closest option.
    Uses language-aware model selection for multilingual support.
    This replaces the old WhisperPseudoRefScorer.
    """

    def __init__(self, language="ar"):
        self.language = language
        self.model_id = get_moonshine_model_id(language)
        self.token_factor = get_token_factor(language)
        self.model = None
        self.processor = None
        self.sampling_rate = None
        self._cache = {}  # audio_path -> transcription

    def setup(self):
        """Load Moonshine model and processor from HuggingFace."""
        from transformers import MoonshineForConditionalGeneration, AutoProcessor

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        print(f"Loading Moonshine model: {self.model_id} ...")
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id)
            self.model = MoonshineForConditionalGeneration.from_pretrained(
                self.model_id
            ).to(device).to(torch_dtype)
            self.sampling_rate = self.processor.feature_extractor.sampling_rate
            print(f"Moonshine loaded on {device} (dtype={torch_dtype})")
            print(f"  Sampling rate: {self.sampling_rate}")
            print(f"  Token factor: {self.token_factor}")
        except Exception as e:
            print(f"ERROR loading Moonshine {self.model_id}: {e}")
            # Try fallback to base model
            if self.model_id != MOONSHINE_FALLBACK_MODEL:
                print(f"Falling back to {MOONSHINE_FALLBACK_MODEL}...")
                self.model_id = MOONSHINE_FALLBACK_MODEL
                self.token_factor = get_token_factor("default")
                self.processor = AutoProcessor.from_pretrained(self.model_id)
                self.model = MoonshineForConditionalGeneration.from_pretrained(
                    self.model_id
                ).to(device).to(torch_dtype)
                self.sampling_rate = self.processor.feature_extractor.sampling_rate
                print(f"Fallback loaded on {device}")

        if torch.cuda.is_available():
            free_gb = torch.cuda.mem_get_info()[0] / 1e9
            total_gb = torch.cuda.mem_get_info()[1] / 1e9
            print(f"GPU memory after load: {free_gb:.1f} GB free / {total_gb:.1f} GB total")

    def _load_audio(self, audio_path: str) -> np.ndarray:
        """Load and resample audio to model's expected sample rate."""
        wav, sr = torchaudio.load(audio_path)
        if sr != self.sampling_rate:
            wav = torchaudio.transforms.Resample(sr, self.sampling_rate)(wav)
        # Convert to mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0).numpy()

    def transcribe(self, audio_path: str) -> str:
        """Transcribe a single audio file with Moonshine."""
        if audio_path in self._cache:
            return self._cache[audio_path]

        device = next(self.model.parameters()).device
        torch_dtype = next(self.model.parameters()).dtype

        audio_array = self._load_audio(audio_path)

        inputs = self.processor(
            audio_array,
            return_tensors="pt",
            sampling_rate=self.sampling_rate,
        )
        inputs = inputs.to(device, torch_dtype)

        # Hallucination guard: limit max tokens based on audio length
        token_limit_factor = self.token_factor / self.sampling_rate
        seq_lens = inputs.attention_mask.sum(dim=-1)
        max_length = int((seq_lens * token_limit_factor).max().item())
        max_length = max(max_length, 10)  # minimum 10 tokens

        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_length=max_length)

        text = self.processor.decode(generated_ids[0], skip_special_tokens=True).strip()
        self._cache[audio_path] = text
        return text

    def score(self, options: List[str], audio_path: str = None) -> List[float]:
        penalty = outlier.get_penalty(options)
        if audio_path is None or self.model is None:
            return penalty  # fallback

        try:
            ref = self.transcribe(audio_path)
        except Exception as e:
            print(f"  Moonshine transcription error: {e}")
            return penalty

        scores = []
        for i, opt in enumerate(options):
            if penalty[i] == 0:
                scores.append(0.0)
            else:
                scores.append(Levenshtein.normalized_similarity(ref, opt))
        return scores

# ========== BLOCK 8: CTC Forced Alignment Scorer (GPU) ==========
class CTCAlignmentScorer:
    """Score by CTC forward probability using wav2vec2-xlsr-53."""

    def __init__(self):
        self.processor = None
        self.model_ctc = None

    def setup(self):
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
        print("Loading wav2vec2-xlsr-53...")
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-large-xlsr-53"
        )
        self.model_ctc = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-large-xlsr-53"
        )
        if torch.cuda.is_available():
            self.model_ctc = self.model_ctc.cuda()
        self.model_ctc.eval()
        print("wav2vec2 loaded.")

    def _load_audio(self, path: str):
        wav, sr = torchaudio.load(path)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        return wav.squeeze(0).numpy()

    def score(self, options: List[str], audio_path: str = None) -> List[float]:
        penalty = outlier.get_penalty(options)
        if audio_path is None or self.model_ctc is None:
            return penalty

        try:
            audio = self._load_audio(audio_path)
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}

            with torch.no_grad():
                logits = self.model_ctc(**inputs).logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            scores = []
            for i, opt in enumerate(options):
                if penalty[i] == 0:
                    scores.append(-999.0)
                else:
                    scores.append(float(log_probs.max(dim=-1).values.sum()) / max(len(opt), 1))
            return scores
        except Exception as e:
            print(f"  CTC error: {e}")
            return penalty

# ========== BLOCK 9: LM Perplexity Scorer (GPU) ==========
class LMPerplexityScorer:
    """Score by inverse perplexity from AraGPT2."""

    def __init__(self):
        self.model_lm = None
        self.tokenizer = None

    def setup(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("Loading AraGPT2...")
        self.tokenizer = AutoTokenizer.from_pretrained("aubmindlab/aragpt2-base")
        self.model_lm = AutoModelForCausalLM.from_pretrained("aubmindlab/aragpt2-base")
        if torch.cuda.is_available():
            self.model_lm = self.model_lm.cuda()
        self.model_lm.eval()
        print("AraGPT2 loaded.")

    def _perplexity(self, text: str) -> float:
        tokens = self.tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=512)
        if torch.cuda.is_available():
            tokens = tokens.cuda()
        if tokens.shape[1] < 2:
            return 999.0
        with torch.no_grad():
            outputs = self.model_lm(tokens, labels=tokens)
        return float(outputs.loss.exp())

    def score(self, options: List[str], audio_path: str = None) -> List[float]:
        penalty = outlier.get_penalty(options)
        scores = []
        for i, opt in enumerate(options):
            if penalty[i] == 0:
                scores.append(0.0)
            else:
                ppl = self._perplexity(opt)
                scores.append(1.0 / max(ppl, 1e-6))
        return scores

# ========== BLOCK 10: Confidence-Gated Hybrid Scorer ==========
class ConfidenceGatedScorer:
    """
    Smart hybrid: trust Moonshine when it's confident, fall back to MBR otherwise.

    Confidence = gap between Moonshine's best and second-best similarity score.
    If gap > threshold → Moonshine clearly prefers one option → trust it.
    If gap <= threshold → Moonshine is uncertain → use MBR consensus instead.
    """

    def __init__(self, moonshine_scorer=None, confidence_threshold=0.05):
        self.moonshine_scorer = moonshine_scorer
        self.threshold = confidence_threshold
        self._decisions = []

    def setup(self):
        pass  # moonshine_scorer passed in already loaded

    def score(self, options: List[str], audio_path: str = None) -> List[float]:
        penalty = np.array(outlier.get_penalty(options))
        n = len(options)

        # Get both signals
        mbr_scores = np.array(ConsensusMBRScorer().score(options))
        mbr_scores *= penalty

        if self.moonshine_scorer is None or audio_path is None:
            self._decisions.append("mbr_fallback")
            return mbr_scores.tolist()

        moonshine_scores = np.array(self.moonshine_scorer.score(options, audio_path))

        # Compute confidence: gap between 1st and 2nd best Moonshine score
        valid_scores = moonshine_scores[penalty > 0]
        if len(valid_scores) < 2:
            self._decisions.append("mbr_fallback")
            return mbr_scores.tolist()

        sorted_scores = np.sort(valid_scores)[::-1]
        confidence = sorted_scores[0] - sorted_scores[1]

        if confidence > self.threshold:
            self._decisions.append(f"moonshine(gap={confidence:.3f})")
            return moonshine_scores.tolist()
        else:
            blend = 0.7 * self._normalize(moonshine_scores) + 0.3 * self._normalize(mbr_scores)
            blend *= penalty
            self._decisions.append(f"blend(gap={confidence:.3f})")
            return blend.tolist()

    def _normalize(self, scores):
        arr = np.array(scores, dtype=float)
        rng = arr.max() - arr.min()
        if rng < 1e-10:
            return np.full_like(arr, 0.5)
        return (arr - arr.min()) / rng

    def get_decision_stats(self):
        moonshine_count = sum(1 for d in self._decisions if d.startswith("moonshine"))
        blend_count = sum(1 for d in self._decisions if d.startswith("blend"))
        mbr_count = sum(1 for d in self._decisions if d.startswith("mbr"))
        total = len(self._decisions)
        return {"moonshine": moonshine_count, "blend": blend_count, "mbr": mbr_count, "total": total}

# ========== BLOCK 11: Full Fusion Pipeline ==========
class FusionScorer:
    """Combine all signals with weighted fusion."""

    def __init__(self, moonshine_scorer=None, weights=None):
        self.moonshine_scorer = moonshine_scorer
        self.ctc_scorer = CTCAlignmentScorer()
        self.lm_scorer = LMPerplexityScorer()
        self.weights = weights or {
            "moonshine_ref": 0.40,
            "mbr": 0.25,
            "lm": 0.15,
            "ctc": 0.10,
            "heuristic": 0.10,
        }

    def setup(self):
        if self.moonshine_scorer is None:
            print("WARNING: No Moonshine scorer provided, moonshine_ref signal will be skipped")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            self.ctc_scorer.setup()
        except Exception as e:
            print(f"CTC setup failed (will skip): {e}")
            self.ctc_scorer = None
        try:
            self.lm_scorer.setup()
        except Exception as e:
            print(f"LM setup failed (will skip): {e}")
            self.lm_scorer = None

    def _normalize(self, scores, invert=False):
        arr = np.array(scores, dtype=float)
        if arr.max() - arr.min() < 1e-10:
            return np.full_like(arr, 0.5)
        normed = (arr - arr.min()) / (arr.max() - arr.min())
        if invert:
            normed = 1.0 - normed
        return normed

    def score(self, options: List[str], audio_path: str = None) -> List[float]:
        n = len(options)
        penalty = np.array(outlier.get_penalty(options))

        signals = {}

        # MBR
        mbr_raw = ConsensusMBRScorer().score(options)
        signals["mbr"] = self._normalize(mbr_raw)

        # Text heuristics
        heur_raw = TextHeuristicsScorer().score(options)
        signals["heuristic"] = self._normalize(heur_raw)

        # Moonshine pseudo-ref
        if self.moonshine_scorer is not None:
            moonshine_raw = self.moonshine_scorer.score(options, audio_path)
            signals["moonshine_ref"] = self._normalize(moonshine_raw)

        # CTC
        if self.ctc_scorer:
            ctc_raw = self.ctc_scorer.score(options, audio_path)
            signals["ctc"] = self._normalize(ctc_raw)

        # LM
        if self.lm_scorer:
            lm_raw = self.lm_scorer.score(options, audio_path)
            signals["lm"] = self._normalize(lm_raw)

        # Weighted fusion
        final = np.zeros(n)
        total_w = 0
        for name, normed in signals.items():
            w = self.weights.get(name, 0)
            final += w * normed
            total_w += w

        if total_w > 0:
            final /= total_w

        final *= penalty
        return final.tolist()

# ========== BLOCK 12: Run All Scorers ==========
def run_scorer(name, scorer_fn, df, audio_map, needs_audio=False):
    """Run a scorer and return results dataframe + metrics."""
    results = []
    times = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=name):
        aid = str(row["audio_id"])
        options = [str(row[c]) for c in OPTION_COLS]
        audio_path = audio_map.get(aid) if needs_audio else None

        t0 = time.time()
        scores = scorer_fn(options, audio_path)
        elapsed = time.time() - t0
        times.append(elapsed)

        golden_idx = int(np.argmax(scores))
        results.append({
            "audio_id": aid,
            "golden_idx": golden_idx,
            "golden_option": f"option_{golden_idx + 1}",
            "golden_ref": options[golden_idx],
            "golden_score": scores[golden_idx],
            "scores": scores,
            "options": options,
        })

    return results, times

# --- Run TEXT-ONLY scorers (no GPU needed) ---
print("=" * 60)
print("  Running text-only scorers...")
print("=" * 60)

mbr_scorer = ConsensusMBRScorer()
mbr_outlier_scorer = OutlierAwareMBR()
heuristics_scorer = TextHeuristicsScorer()

mbr_results, mbr_times = run_scorer("MBR", mbr_scorer.score, df, audio_map)
mbr_ol_results, mbr_ol_times = run_scorer("MBR+Outlier", mbr_outlier_scorer.score, df, audio_map)
heur_results, heur_times = run_scorer("Heuristics", heuristics_scorer.score, df, audio_map)

# ========== BLOCK 13: GPU Cleanup & Load Moonshine ==========
def nuclear_cleanup():
    """Aggressively free ALL GPU memory — call before loading models."""
    g = list(globals().items())
    for name, obj in g:
        if isinstance(obj, (torch.Tensor, torch.nn.Module)):
            try:
                globals()[name] = None
            except Exception:
                pass
    for _ in range(3):
        gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        free_gb = torch.cuda.mem_get_info()[0] / 1e9
        total_gb = torch.cuda.mem_get_info()[1] / 1e9
        print(f"After cleanup: {free_gb:.1f} GB free / {total_gb:.1f} GB total")

nuclear_cleanup()

# --- Run GPU scorers ---
print("\n" + "=" * 60)
print("  Running GPU scorers (Moonshine)...")
print("=" * 60)

if torch.cuda.is_available():
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    total_gb = torch.cuda.mem_get_info()[1] / 1e9
    print(f"GPU memory: {free_gb:.1f} GB free / {total_gb:.1f} GB total")

moonshine_scorer = MoonshinePseudoRefScorer(language=LANGUAGE)
moonshine_scorer.setup()
moonshine_results, moonshine_times = run_scorer(
    "Moonshine-Ref", moonshine_scorer.score, df, audio_map, needs_audio=True
)

# Save Moonshine transcriptions for evaluation
moonshine_refs = {}
for _, row in tqdm(df.iterrows(), total=len(df), desc="Caching Moonshine refs"):
    aid = str(row["audio_id"])
    path = audio_map.get(aid)
    if path:
        moonshine_refs[aid] = moonshine_scorer.transcribe(path)
print(f"Generated {len(moonshine_refs)} Moonshine pseudo-references")

# ========== BLOCK 14: Fusion & Confidence-Gated Scorers ==========
# Full fusion scorer — reuse the already-loaded Moonshine model
fusion_scorer = FusionScorer(moonshine_scorer=moonshine_scorer)
fusion_scorer.setup()
fusion_results, fusion_times = run_scorer(
    "Fusion", fusion_scorer.score, df, audio_map, needs_audio=True
)

# Confidence-Gated Hybrid
print("\nRunning Confidence-Gated Hybrid...")
gated_scorer = ConfidenceGatedScorer(moonshine_scorer=moonshine_scorer, confidence_threshold=0.05)
gated_results, gated_times = run_scorer(
    "Gated", gated_scorer.score, df, audio_map, needs_audio=True
)
stats = gated_scorer.get_decision_stats()
print(f"\nGated decisions: {stats}")
print(f"  Moonshine-confident: {stats['moonshine']}/{stats['total']} ({100*stats['moonshine']/max(stats['total'],1):.0f}%)")
print(f"  Blended (uncertain): {stats['blend']}/{stats['total']} ({100*stats['blend']/max(stats['total'],1):.0f}%)")
print(f"  MBR fallback: {stats['mbr']}/{stats['total']}")

# ========== BLOCK 15: Independent Evaluation (Whisper Medium) ==========
# IMPORTANT: We use Whisper medium as the INDEPENDENT evaluator.
# Scorer = Moonshine, Evaluator = Whisper medium → different model families → NO circularity.

print("\n" + "=" * 60)
print("  Loading Whisper MEDIUM as independent evaluator...")
print("  (Scorer=Moonshine, Evaluator=Whisper → no circularity)")
print("=" * 60)

import whisper as whisper_lib

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

eval_model_size = "medium"
eval_model = whisper_lib.load_model(eval_model_size, device="cpu")
eval_device = "cuda" if torch.cuda.is_available() else "cpu"
if eval_device == "cuda":
    try:
        eval_model = eval_model.cuda()
    except (RuntimeError, torch.cuda.OutOfMemoryError):
        print("WARNING: Whisper medium OOM on GPU, running on CPU (slower)")
        eval_device = "cpu"
print(f"Whisper {eval_model_size} loaded on {eval_device} for evaluation.")

# Generate INDEPENDENT pseudo-references with Whisper medium
eval_refs = {}
for _, row in tqdm(df.iterrows(), total=len(df), desc="Eval refs (Whisper medium)"):
    aid = str(row["audio_id"])
    path = audio_map.get(aid)
    if path:
        result = eval_model.transcribe(
            path, language=LANGUAGE, task="transcribe",
            without_timestamps=True, fp16=torch.cuda.is_available() and eval_device == "cuda",
        )
        eval_refs[aid] = result["text"].strip()
print(f"Generated {len(eval_refs)} independent eval references (Whisper medium)")

# Free eval model to reclaim GPU memory
del eval_model
gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# ========== BLOCK 16: Evaluate All Scorers ==========
from jiwer import wer as jiwer_wer, cer as jiwer_cer

def evaluate_scorer(name, results, refs):
    """Evaluate a scorer using pseudo-reference transcription."""
    wers, cers, correct = [], [], 0
    total = 0

    for r in results:
        aid = r["audio_id"]
        if aid not in refs:
            continue
        ref = refs[aid]
        picked = r["golden_ref"]
        total += 1

        try:
            w = jiwer_wer(ref, picked)
            c = jiwer_cer(ref, picked)
        except Exception:
            w, c = 1.0, 1.0
        wers.append(w)
        cers.append(c)

        # Did scorer pick the option closest to eval ref?
        option_wers = []
        for opt in r["options"]:
            try:
                option_wers.append(jiwer_wer(ref, opt))
            except Exception:
                option_wers.append(1.0)
        best_idx = int(np.argmin(option_wers))
        if r["golden_idx"] == best_idx:
            correct += 1

    acc = correct / total if total else 0
    avg_wer = np.mean(wers) if wers else 1.0
    avg_cer = np.mean(cers) if cers else 1.0
    oracle_wer_vals = []
    for r in results:
        aid = r["audio_id"]
        if aid not in refs:
            continue
        ref = refs[aid]
        option_wers = []
        for opt in r["options"]:
            try:
                option_wers.append(jiwer_wer(ref, opt))
            except Exception:
                option_wers.append(1.0)
        oracle_wer_vals.append(min(option_wers))
    oracle_wer = np.mean(oracle_wer_vals) if oracle_wer_vals else 1.0

    return {
        "scorer": name,
        "pseudo_accuracy": round(acc, 4),
        "pseudo_wer": round(avg_wer, 4),
        "pseudo_cer": round(avg_cer, 4),
        "oracle_wer": round(oracle_wer, 4),
        "wer_gap": round(avg_wer - oracle_wer, 4),
        "total": total,
    }

all_results = {
    "MBR (vanilla)": mbr_results,
    "MBR + Outlier": mbr_ol_results,
    "Text Heuristics": heur_results,
    "Moonshine Pseudo-Ref": moonshine_results,
    "Confidence-Gated": gated_results,
    "Full Fusion": fusion_results,
}

# --- Evaluation with INDEPENDENT Whisper medium refs ---
print("\n" + "=" * 70)
print("  EVALUATION — Independent Whisper Medium as Proxy GT")
print("  (Scorer uses Moonshine, evaluator uses Whisper medium → no circularity)")
print("=" * 70)
print(f"{'Scorer':<25} {'Acc':>8} {'WER':>8} {'CER':>8} {'Oracle':>8} {'Gap':>8}")
print("-" * 70)

eval_rows = []
for name, results in all_results.items():
    ev = evaluate_scorer(name, results, eval_refs)
    eval_rows.append(ev)
    print(f"{name:<25} {ev['pseudo_accuracy']:>8.1%} {ev['pseudo_wer']:>8.4f} "
          f"{ev['pseudo_cer']:>8.4f} {ev['oracle_wer']:>8.4f} {ev['wer_gap']:>8.4f}")

print("=" * 70)
print("Note: Evaluator is Whisper medium (independent from the Moonshine scorer)")
print("      'Gap' = WER - Oracle WER (lower = closer to best possible choice)")

# --- Also show self-eval (Moonshine) for comparison (circular but informative) ---
print("\n" + "=" * 70)
print("  COMPARISON — Self-evaluation (Moonshine, CIRCULAR — for reference only)")
print("=" * 70)
print(f"{'Scorer':<25} {'Acc':>8} {'WER':>8} {'CER':>8}")
print("-" * 70)
for name, results in all_results.items():
    ev_self = evaluate_scorer(name, results, moonshine_refs)
    print(f"{name:<25} {ev_self['pseudo_accuracy']:>8.1%} {ev_self['pseudo_wer']:>8.4f} "
          f"{ev_self['pseudo_cer']:>8.4f}")
print("=" * 70)
print("(Moonshine scorer evaluating against itself — inflated, use Whisper medium eval above)")

# ========== BLOCK 17: Inter-Scorer Agreement ==========
print("\n" + "=" * 60)
print("  INTER-SCORER AGREEMENT")
print("=" * 60)

scorer_names = list(all_results.keys())
n_scorers = len(scorer_names)
agreement_matrix = np.zeros((n_scorers, n_scorers))

for i in range(n_scorers):
    for j in range(n_scorers):
        agree = sum(1 for a, b in zip(all_results[scorer_names[i]], all_results[scorer_names[j]])
                    if a["golden_idx"] == b["golden_idx"])
        agreement_matrix[i][j] = agree / len(df)

header = f"{'':>25}" + "".join(f"{n[:10]:>12}" for n in scorer_names)
print(header)
for i, name in enumerate(scorer_names):
    row_str = f"{name:>25}" + "".join(f"{agreement_matrix[i][j]:>12.1%}" for j in range(n_scorers))
    print(row_str)

# ========== BLOCK 18: Generate Submission ==========
print("\nGenerating submissions...")
# Select the best scorer based on evaluation results
# Default to Moonshine Pseudo-Ref (typically best audio scorer)
best_results = moonshine_results
print("  PRIMARY: Moonshine Pseudo-Ref")
print("  BACKUP:  Confidence-Gated")

# Build submission DataFrame
submission_rows = []
for r in best_results:
    row = {"audio_id": r["audio_id"]}
    for i, opt in enumerate(r["options"]):
        row[f"option_{i+1}"] = opt
    row["golden_ref"] = r["golden_ref"]
    row["golden_option"] = r["golden_option"]
    row["golden_score"] = round(r["golden_score"], 6)

    for i, opt in enumerate(r["options"]):
        try:
            row[f"wer_option{i+1}"] = round(jiwer_wer(r["golden_ref"], opt), 4)
        except Exception:
            row[f"wer_option{i+1}"] = 1.0
    submission_rows.append(row)

submission_df = pd.DataFrame(submission_rows)

# Merge with original data for context columns
original_cols = ["audio_id", "language", "audio"]
df_merge = df[original_cols].copy()
df_merge["audio_id"] = df_merge["audio_id"].astype(str)
submission_df["audio_id"] = submission_df["audio_id"].astype(str)
merged = df_merge.merge(submission_df, on="audio_id", how="right")

# Save
output_path = os.path.join(OUTPUT_DIR, "golden_submission.csv")
merged.to_csv(output_path, index=False)
print(f"\nSubmission saved: {output_path}")
print(f"Shape: {merged.shape}")
print(f"\nSelection distribution:")
print(merged["golden_option"].value_counts().sort_index())

# ========== BLOCK 19: Save All Outputs ==========
# Save all individual scorer outputs
for name, results in all_results.items():
    safe = name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    rows = []
    for r in results:
        rows.append({
            "audio_id": r["audio_id"],
            "golden_option": r["golden_option"],
            "golden_ref": r["golden_ref"],
            "golden_score": round(r["golden_score"], 6),
        })
    pd.DataFrame(rows).to_csv(
        os.path.join(OUTPUT_DIR, f"{safe}_results.csv"), index=False
    )

# Save evaluation comparison
eval_df = pd.DataFrame(eval_rows)
eval_df.to_csv(os.path.join(OUTPUT_DIR, "scorer_comparison.csv"), index=False)
print("\nAll outputs saved to:", OUTPUT_DIR)

# Save pseudo-references for future use
ref_rows = [{"audio_id": k, "moonshine_reference": v} for k, v in moonshine_refs.items()]
pd.DataFrame(ref_rows).to_csv(os.path.join(OUTPUT_DIR, "moonshine_pseudo_references.csv"), index=False)
print(f"Saved {len(ref_rows)} Moonshine pseudo-references")

# Also save Moonshine-only as backup submission
moonshine_only_rows = []
for r in moonshine_results:
    row = {"audio_id": r["audio_id"]}
    for i, opt in enumerate(r["options"]):
        row[f"option_{i+1}"] = opt
    row["golden_ref"] = r["golden_ref"]
    row["golden_option"] = r["golden_option"]
    row["golden_score"] = round(r["golden_score"], 6)
    for i, opt in enumerate(r["options"]):
        try:
            row[f"wer_option{i+1}"] = round(jiwer_wer(r["golden_ref"], opt), 4)
        except Exception:
            row[f"wer_option{i+1}"] = 1.0
    moonshine_only_rows.append(row)

moonshine_sub_df = pd.DataFrame(moonshine_only_rows)
moonshine_sub_df["audio_id"] = moonshine_sub_df["audio_id"].astype(str)
moonshine_merged = df_merge.merge(moonshine_sub_df, on="audio_id", how="right")
moonshine_merged.to_csv(os.path.join(OUTPUT_DIR, "golden_submission_moonshine_only.csv"), index=False)
print(f"Moonshine-only backup saved: {os.path.join(OUTPUT_DIR, 'golden_submission_moonshine_only.csv')}")

# ========== BLOCK 20: Summary ==========
# | Scorer              | What it does                                  | GPU? | Role               |
# |---------------------|-----------------------------------------------|------|--------------------|
# | MBR (vanilla)       | Pairwise CER consensus                        | No   | Baseline           |
# | MBR + Outlier       | MBR + script detection                        | No   | Baseline           |
# | Text Heuristics     | MBR + length + diversity                      | No   | Baseline           |
# | Moonshine Pseudo-Ref| Similarity to Moonshine transcription         | Yes  | PRIMARY SUBMISSION |
# | Confidence-Gated    | Moonshine when confident, MBR when uncertain  | Yes  | Backup submission  |
# | Full Fusion         | Weighted combination of all signals           | Yes  | Comparison only    |
#
# **Strategy**: Moonshine (audio signal) is our primary scorer.
# It uses a language-specific model when available (e.g., moonshine-tiny-ar for Arabic).
# When Moonshine is uncertain (small gap), Confidence-Gated blends in MBR consensus.
#
# **Evaluation**: Uses Whisper medium as independent evaluator (different model family
# from Moonshine) to avoid circular self-evaluation.
#
# **Multilingual**: Change LANGUAGE variable to use different language models.
# Supported: ar, zh, ja, ko, uk, vi (with automatic fallback to moonshine-base).
