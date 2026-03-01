"""
Configuration constants for the GoldenASR pipeline.
"""

import torch

# --- Device ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- ASR Models ---
WHISPER_MODEL = "large-v3"

# --- Regime Detection ---
DIVERSITY_THRESHOLD = 0.3

# --- Paths (defaults, override via CLI or environment) ---
OUTPUT_CSV = "golden_transcriptions_output.csv"
AUDIO_DIR = "audio_files"

# --- Download Settings ---
DOWNLOAD_WORKERS = 10
DOWNLOAD_TIMEOUT = 120

# --- Whisper Settings ---
WHISPER_BEAM_SIZE = 5
WHISPER_BEST_OF = 5
WHISPER_TEMPERATURE = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
WHISPER_NO_SPEECH_THRESHOLD = 0.6
WHISPER_COMPRESSION_RATIO_THRESHOLD = 2.4

# --- SeamlessM4T Language Mapping ---
LANG_TO_SEAMLESS = {
    "arabic": "arb",
    "Arabic_SA": "arb",
    "arb": "arb",
    "english": "eng",
    "English": "eng",
    "eng": "eng",
    "hindi": "hin",
    "Hindi": "hin",
    "hin": "hin",
    "french": "fra",
    "spanish": "spa",
    "german": "deu",
}

# --- Grid Search Space ---
GRID = {
    "whisper": [0.0, 0.1, 0.2, 0.3, 0.5, 0.7],
    "seamless": [0.0, 0.1, 0.2, 0.3, 0.5, 0.7],
    "consensus": [-0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2],
    "fluency": [-0.3, -0.2, -0.1, 0.0],
    "quality": [0.0, 0.1, 0.2, 0.3, 0.5],
    "length": [0.0, 0.05, 0.1],
}

# --- Default Fallback Weights ---
DEFAULT_SIMILAR_WEIGHTS = {
    "whisper": 0,
    "seamless": 0.1,
    "consensus": -0.3,
    "fluency": -0.2,
    "quality": 0.1,
    "length": 0,
}

DEFAULT_DIVERSE_WEIGHTS = {
    "whisper": 0,
    "seamless": 0.3,
    "consensus": 0.2,
    "fluency": 0,
    "quality": 0,
    "length": 0.05,
}
