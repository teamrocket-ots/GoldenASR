"""
SeamlessM4T v2-large transcription backend.
"""

import gc
import os

import torch
import torchaudio
from tqdm.auto import tqdm
from transformers import AutoProcessor, SeamlessM4Tv2ForSpeechToText

from golden_asr.config import DEVICE, LANG_TO_SEAMLESS


def load_seamless_model(device=None):
    """Load the SeamlessM4T v2-large model and processor.

    Args:
        device: Device string. Defaults to ``config.DEVICE``.

    Returns:
        tuple: (model, processor).
    """
    device = device or DEVICE
    print("Loading SeamlessM4T v2-large...")
    processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
    model = SeamlessM4Tv2ForSpeechToText.from_pretrained(
        "facebook/seamless-m4t-v2-large"
    )
    model = model.to(device)
    model.eval()
    print("SeamlessM4T loaded.")
    return model, processor


def transcribe_single(model, processor, filepath, lang_hint="Arabic_SA", device=None):
    """Transcribe a single audio file using SeamlessM4T.

    Args:
        model: Loaded SeamlessM4T model.
        processor: Matching AutoProcessor.
        filepath: Path to the audio file.
        lang_hint: Language hint for target language selection.
        device: Device string. Defaults to ``config.DEVICE``.

    Returns:
        str: Transcribed text.
    """
    device = device or DEVICE
    try:
        waveform, sr = torchaudio.load(filepath)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        inputs = processor(
            audio=waveform.squeeze(0).numpy(),
            sampling_rate=16000,
            return_tensors="pt",
        ).to(device)

        tgt = LANG_TO_SEAMLESS.get(lang_hint, "arb")

        with torch.no_grad():
            output_tokens = model.generate(**inputs, tgt_lang=tgt)
        text = processor.decode(output_tokens[0].tolist(), skip_special_tokens=True)
        return text.strip()
    except Exception as exc:
        print(f"  SeamlessM4T error for {filepath}: {exc}")
        return ""


def transcribe_dataset(df, audio_paths):
    """Transcribe all audio files in the dataset with SeamlessM4T.

    The model is loaded, used for all samples, and then unloaded to free
    GPU memory.

    Args:
        df: Dataset DataFrame with ``audio_id`` and ``language`` columns.
        audio_paths: Dict mapping audio_id to local file path.

    Returns:
        dict: Mapping of audio_id to transcribed text.
    """
    model, processor = load_seamless_model()

    seamless_texts = {}

    print("Transcribing with SeamlessM4T v2...")
    for _, row in tqdm(df.iterrows(), total=len(df), desc="SeamlessM4T"):
        aid = row["audio_id"]
        fp = audio_paths.get(aid, "")
        lang = row.get("language", "Arabic_SA")
        if fp and os.path.exists(str(fp)):
            seamless_texts[aid] = transcribe_single(
                model, processor, fp, lang_hint=lang
            )
        else:
            seamless_texts[aid] = ""

    print("SeamlessM4T done.")

    # Free GPU memory
    del model, processor
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("SeamlessM4T model unloaded.")

    return seamless_texts
