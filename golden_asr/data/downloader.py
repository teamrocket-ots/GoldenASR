"""
Audio file downloading utilities for the GoldenASR pipeline.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm.auto import tqdm

from golden_asr.config import AUDIO_DIR, DOWNLOAD_WORKERS, DOWNLOAD_TIMEOUT


def _download_single(audio_id, url, audio_dir, timeout):
    """Download a single audio file.

    Args:
        audio_id: Unique identifier of the audio sample.
        url: Remote URL to download from.
        audio_dir: Local directory to write the audio file.
        timeout: HTTP request timeout in seconds.

    Returns:
        tuple: (audio_id, filepath, success).
    """
    filepath = os.path.join(audio_dir, f"audio_{audio_id}.wav")
    if os.path.exists(filepath):
        return (audio_id, filepath, True)
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        with open(filepath, "wb") as f:
            f.write(resp.content)
        return (audio_id, filepath, True)
    except Exception as exc:
        print(f"  FAILED audio_id={audio_id}: {exc}")
        return (audio_id, filepath, False)


def download_all_audio(df, audio_dir=None, max_workers=None, timeout=None):
    """Download audio files for every row in the dataset.

    Args:
        df: DataFrame with ``audio_id`` and ``audio`` (URL) columns.
        audio_dir: Directory to store downloaded files.
            Defaults to ``config.AUDIO_DIR``.
        max_workers: Number of parallel download threads.
            Defaults to ``config.DOWNLOAD_WORKERS``.
        timeout: Per-request timeout in seconds.
            Defaults to ``config.DOWNLOAD_TIMEOUT``.

    Returns:
        dict: Mapping of audio_id to local filepath for successful downloads.
    """
    audio_dir = audio_dir or AUDIO_DIR
    max_workers = max_workers or DOWNLOAD_WORKERS
    timeout = timeout or DOWNLOAD_TIMEOUT

    os.makedirs(audio_dir, exist_ok=True)

    audio_paths = {}
    print(f"Downloading {len(df)} audio files...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                _download_single, row["audio_id"], str(row["audio"]), audio_dir, timeout
            ): row["audio_id"]
            for _, row in df.iterrows()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            audio_id, filepath, success = future.result()
            if success:
                audio_paths[audio_id] = filepath

    return audio_paths
