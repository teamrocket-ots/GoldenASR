"""
Hard quality filters for candidate transcription options.

Golden transcriptions are always pure verbatim text. They never contain
speaker labels, stage directions, or scene markers. These filters
eliminate clearly non-golden candidates before scoring.
"""

import re


def is_script_option(text):
    """Return True if the text looks like a full screenplay script (>1000 chars).

    Args:
        text: Candidate transcription text.

    Returns:
        bool
    """
    return isinstance(text, str) and len(text) > 1000


def has_speaker_labels(text):
    """Return True if the text contains speaker label patterns.

    Detects patterns like ``[Speaker:]`` and ``ArabicName:``.

    Args:
        text: Candidate transcription text.

    Returns:
        bool
    """
    if not isinstance(text, str):
        return False
    if re.search(r"\[[^\]]*:\]", text):
        return True
    if re.search(r"(?:^|[\n])[\u0600-\u06FF]+\s*:", text):
        return True
    return False


def has_stage_directions(text):
    """Return True if the text contains stage directions in parentheses.

    Looks for ``(text with 3+ chars)`` patterns.

    Args:
        text: Candidate transcription text.

    Returns:
        bool
    """
    if not isinstance(text, str):
        return False
    return bool(re.search(r"\([^)]{3,}\)", text))


def has_scene_markers(text):
    """Return True if the text contains scene/location markers.

    Args:
        text: Candidate transcription text.

    Returns:
        bool
    """
    if not isinstance(text, str):
        return False
    markers = [
        "\u0627\u0644\u0645\u0634\u0647\u062F",  # Arabic "scene"
        "\u0627\u0644\u0645\u0643\u0627\u0646:",  # Arabic "location:"
        "Scene ",
    ]
    return any(m in text for m in markers)


def compute_quality_penalty(text):
    """Compute a quality penalty score for a candidate transcription.

    Higher penalty means more annotation artifacts, making it less likely
    to be the golden transcription.

    Args:
        text: Candidate transcription text.

    Returns:
        float: Penalty in [0.0, 1.0].
    """
    if not isinstance(text, str) or len(text.strip()) < 3:
        return 1.0
    penalty = 0.0
    penalty += len(re.findall(r"\[[^\]]*:\]", text)) * 0.15
    penalty += len(re.findall(r"\([^)]{3,}\)", text)) * 0.12
    penalty += len(re.findall(r"[\u0600-\u06FF]+\s*:", text)) * 0.05
    if has_scene_markers(text):
        penalty += 0.2
    penalty += text.count("{") * 0.02
    return min(penalty, 1.0)


def passes_hard_filter(text):
    """Return True if the candidate passes all hard filters.

    A candidate passes if it is not a script, has no speaker labels,
    no stage directions, and no scene markers.

    Args:
        text: Candidate transcription text.

    Returns:
        bool
    """
    return not any([
        is_script_option(text),
        has_speaker_labels(text),
        has_stage_directions(text),
        has_scene_markers(text),
    ])
