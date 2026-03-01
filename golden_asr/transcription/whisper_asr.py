"""
Whisper large-v3 transcription backend.
"""

import gc
import os

import torch
import whisper
from tqdm.auto import tqdm

from golden_asr.config import (
    DEVICE,
    WHISPER_BEAM_SIZE,
    WHISPER_BEST_OF,
    WHISPER_COMPRESSION_RATIO_THRESHOLD,
    WHISPER_MODEL,
    WHISPER_NO_SPEECH_THRESHOLD,
    WHISPER_TEMPERATURE,
)


def load_whisper_model(model_name=None, device=None):
    """Load the Whisper model.

    Args:
        model_name: Whisper model name. Defaults to ``config.WHISPER_MODEL``.
        device: Device string. Defaults to ``config.DEVICE``.

    Returns:
        whisper model instance.
    """
    model_name = model_name or WHISPER_MODEL
    device = device or DEVICE
    print(f"Loading Whisper {model_name} on {device}...")
    model = whisper.load_model(model_name, device=device)
    print("Whisper model loaded.")
    return model


def transcribe_single(model, filepath):
    """Transcribe a single audio file using Whisper.

    Args:
        model: Loaded Whisper model.
        filepath: Path to the audio file.

    Returns:
        tuple: (transcribed_text, detected_language).
    """
    try:
        result = model.transcribe(
            filepath,
            language=None,
            task="transcribe",
            beam_size=WHISPER_BEAM_SIZE,
            best_of=WHISPER_BEST_OF,
            temperature=WHISPER_TEMPERATURE,
            condition_on_previous_text=True,
            no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
            compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
        )
        return result["text"].strip(), result.get("language", "unknown")
    except Exception as exc:
        print(f"  Whisper error: {exc}")
        return "", "error"


def transcribe_dataset(df, audio_paths):
    """Transcribe all audio files in the dataset with Whisper.

    The Whisper model is loaded, used for all samples, and then unloaded
    to free GPU memory.

    Args:
        df: Dataset DataFrame with ``audio_id`` column.
        audio_paths: Dict mapping audio_id to local file path.

    Returns:
        tuple: (whisper_texts dict, detected_langs dict).
    """
    model = load_whisper_model()

    whisper_texts = {}
    detected_langs = {}

    print("Transcribing with Whisper large-v3...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Whisper"):
        aid = row["audio_id"]
        fp = audio_paths.get(aid, "")
        if fp and os.path.exists(str(fp)):
            text, lang = transcribe_single(model, fp)
            whisper_texts[aid] = text
            detected_langs[aid] = lang
        else:
            whisper_texts[aid] = ""
            detected_langs[aid] = "missing"

    print(f"Whisper done. Detected languages: {dict(zip(*reversed(list(zip(*sorted(detected_langs.items()))))))}"[:200])

    # Free GPU memory
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Whisper model unloaded.")

    return whisper_texts, detected_langs
