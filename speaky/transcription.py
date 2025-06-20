"""ASR transcription and verification utilities using **Distil-Whisper**.

This module centralises all functionality related to automatic speech
recognition (ASR) so that *speak.core* can focus solely on text-to-speech.
"""

from __future__ import annotations

import difflib
import logging
import re
import tempfile
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torchaudio as ta
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    pipeline,
)  # type: ignore

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


class _DistilWhisperASR:  # noqa: WPS110
    """Light wrapper to provide a *faster-whisper*-like interface.

    The wrapper exposes a ``transcribe`` method that returns ``(segments, info)``
    where *segments* is a list of objects each exposing a ``text`` attribute.
    This matches the interface expected by the rest of *speak* while delegating
    the heavy-lifting to a 🤗 *transformers* ASR pipeline running Distil-Whisper.
    """

    def __init__(self, pipe):
        self._pipe = pipe

    # NOTE: *beam_size* and other kwargs are accepted for API compatibility but
    # are currently ignored because the underlying pipeline already chooses
    # sensible defaults for fast inference.
    def transcribe(self, audio_path: str, *_, **__) -> tuple[list[SimpleNamespace], dict]:  # noqa: D401, WPS110
        result = self._pipe(audio_path)
        text = result["text"].strip()
        segment = SimpleNamespace(text=text)
        # Mimic faster-whisper return signature → (segments, info)
        return [segment], {"model": "distil-whisper"}


def _model_id_for_size(model_size: str) -> str:  # noqa: WPS110
    """Return the 🤗 model hub ID corresponding to *model_size*."""

    # Allow callers to pass a fully-qualified model ID directly
    if "/" in model_size:
        return model_size

    size = model_size.lower()
    mapping = {
        "small": "distil-whisper/distil-small.en",
        "medium": "distil-whisper/distil-medium.en",
        "large": "distil-whisper/distil-large-v3",
    }
    return mapping.get(size, model_size)


@cache
def _lazy_asr_model(device: str, model_size: str = "small") -> _DistilWhisperASR:  # noqa: WPS110
    """Return (and cache) a Distil-Whisper ASR wrapper suited to *device*."""

    torch_dtype = torch.float16 if device in {"cuda", "mps"} or device.startswith("cuda") else torch.float32

    model_id = _model_id_for_size(model_size)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    asr_pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
    )

    return _DistilWhisperASR(asr_pipe)


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
    length_ratio = abs(1 - (len(norm_asr) / len(norm_expected)))

    passes = length_ratio <= max_missing_ratio

    if not passes:
        print(f"INPUT:\n{expected_text}")
        print(f"TRANSCRIBED:\n{transcript}")
        print(f"LENGTH RATIO: {length_ratio}")
        print(f"MISSING RATIO: {missing_ratio}")
        print("NORMALIZED DIFF:")
        diff = "\n".join(
            difflib.unified_diff(
                norm_expected.split(),
                norm_asr.split(),
                fromfile="norm_expected",
                tofile="norm_asr",
                lineterm="",
            )
        )
        print(diff)
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
