"""ASR transcription and verification utilities using faster-whisper.

This module centralises all functionality related to automatic speech
recognition (ASR) so that *speak.core* can focus solely on text-to-speech.
"""

from __future__ import annotations

import re
import tempfile
from functools import cache
from pathlib import Path
from typing import Any
import logging
import difflib

import torch
import torchaudio as ta
from faster_whisper import WhisperModel  # type: ignore

__all__ = [
    "_chunk_passes_asr",  # Verification helper
    "_lazy_asr_model",  # Cached model accessor
    "_normalize_for_compare",  # Used internally + by tests
    "transcribe",  # Public helper
]

# ---------------------------------------------------------------------------
#   Normalisation helper (public)
# ---------------------------------------------------------------------------


_NORM_RE = re.compile(r"[^a-z0-9\s]+")


def _normalize_for_compare(text: str) -> str:
    """Return a lightly normalised representation of *text*.

    1. Lower-case
    2. Strip punctuation / non-alphanumerics
    3. Collapse repeated whitespace
    """

    text = text.lower()
    text = _NORM_RE.sub(" ", text)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
#   Lazy Whisper model loader
# ---------------------------------------------------------------------------


@cache
def _lazy_asr_model(device: str, model_size: str = "small") -> WhisperModel:  # noqa: WPS110
    """Return (and cache) a *faster-whisper* model suited to *device*."""

    compute_type = "int8" if device == "cpu" else "float16"
    return WhisperModel(model_size, device=device, compute_type=compute_type)


# ---------------------------------------------------------------------------
#   Stand-alone transcription helper
# ---------------------------------------------------------------------------


def _detect_device(preferred: str | None = None) -> str:  # noqa: WPS110
    """Simple device helper copied to avoid circular import."""

    if preferred:
        return preferred.lower()
    if torch.cuda.is_available():  # type: ignore[attr-defined]
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def transcribe(
    audio_path: str | Path,
    *,
    device: str | None = None,
    model_size: str = "small",
    beam_size: int = 5,
    **whisper_kwargs: Any,
) -> tuple[list[Any], Any]:
    """Transcribe *audio_path* returning ``(segments, info)``.

    The return signature mirrors *faster-whisper* for familiarity.
    """

    device_str = _detect_device(device)
    model = _lazy_asr_model(device_str, model_size=model_size)
    return model.transcribe(str(audio_path), beam_size=beam_size, **whisper_kwargs)


# ---------------------------------------------------------------------------
#   Verification helper for TTS chunks
# ---------------------------------------------------------------------------


def _chunk_passes_asr(
    wav: torch.Tensor,
    sr: int,
    expected_text: str,
    device: str,
    *,
    max_missing_ratio: float,
    model_size: str = "small",
) -> bool:
    """Return *True* iff ASR contains *most* words from *expected_text*."""

    # --- Persist audio to a temporary WAV file ---------------------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir) / "chunk.wav"
        ta.save(str(tmp_path), wav.detach().cpu(), sr)

        segments, _ = _lazy_asr_model(device, model_size=model_size).transcribe(str(tmp_path), beam_size=5)
        transcript = " ".join(seg.text for seg in segments)

    # --- Compare texts ---------------------------------------------------------
    norm_expected = _normalize_for_compare(expected_text)
    norm_asr = _normalize_for_compare(transcript)

    if not norm_expected:
        return True  # Degenerate - cannot fail

    expected_words = norm_expected.split()
    asr_words = set(norm_asr.split())

    missing = [w for w in expected_words if w not in asr_words]
    missing_ratio = len(missing) / len(expected_words)

    passes = missing_ratio <= max_missing_ratio

    if not passes:
        logger = logging.getLogger(__name__)
        diff = "\n".join(
            difflib.unified_diff(
                expected_text.split(),
                transcript.split(),
                fromfile="expected",
                tofile="transcribed",
                lineterm="",
            )
        )
        logger.info("ASR diff (expected vs transcribed):\n%s", diff)

    return passes
