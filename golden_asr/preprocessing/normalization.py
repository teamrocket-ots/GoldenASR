"""
Arabic-optimized text normalization for fair WER comparison.
"""

import re


def normalize_arabic(text):
    """Normalize Arabic-specific characters.

    Removes diacritics, normalizes alef variants, converts taa marbuta
    to haa, alef maksura to yaa, and strips tatweel.

    Args:
        text: Raw Arabic text.

    Returns:
        str: Normalized Arabic text.
    """
    if not text or not isinstance(text, str):
        return ""
    # Remove diacritics
    text = re.sub(r"[\u0617-\u061A\u064B-\u0652\u0670]", "", text)
    # Normalize alef variants
    text = re.sub(r"[\u0625\u0623\u0622\u0671]", "\u0627", text)
    # Taa marbuta to haa
    text = text.replace("\u0629", "\u0647")
    # Alef maksura to yaa
    text = text.replace("\u0649", "\u064A")
    # Remove tatweel
    text = text.replace("\u0640", "")
    return text


def normalize_text(text):
    """Universal text normalization for transcription comparison.

    Removes stage directions, speaker labels, annotations, scene markers,
    punctuation, and applies Arabic-specific normalization. The result is
    lowercased and whitespace-collapsed.

    Args:
        text: Raw transcription text.

    Returns:
        str: Cleaned, normalized text ready for WER/CER computation.
    """
    if not text or not isinstance(text, str):
        return ""
    text = str(text).strip()
    # Remove stage directions
    text = re.sub(r"\([^)]*\)", " ", text)
    # Remove [Speaker:]
    text = re.sub(r"\[[^\]]*:\]", " ", text)
    # Remove [annotations]
    text = re.sub(r"\[[^\]]*\]", " ", text)
    # Scene markers (Arabic)
    text = re.sub(
        r"\u0627\u0644\u0645\u0634\u0647\u062F\s*[\u0660-\u0669\d]+", " ", text
    )
    # Location markers (Arabic)
    text = re.sub(r"\u0627\u0644\u0645\u0643\u0627\u0646\s*:", " ", text)
    # Scene markers (English)
    text = re.sub(r"Scene\s*\d+", " ", text, flags=re.IGNORECASE)
    # Remove {silence} etc.
    text = re.sub(r"[\[\{][^}\]]*[\]\}]", " ", text)
    # Remove punctuation
    text = re.sub(
        r"[\u061F!\u060C\u061B:.\u2026\-\u2013\u2014\"\'\'\`\(\)\[\]\{\}"
        r"\u00AB\u00BB\u201C\u201D\u2018\u2019,;!?\\\\/]",
        " ",
        text,
    )
    # Arabic-specific normalization
    text = normalize_arabic(text)
    # Collapse whitespace and lowercase
    text = re.sub(r"\s+", " ", text).strip().lower()
    return text
