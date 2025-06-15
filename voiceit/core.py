"""Core synthesis utilities for the *voiceit* package.

This module contains **all heavy-lifting logic** (device detection, model
loading, text chunking, audio concatenation, etc.).  Keeping the
implementation here lets the public API (and tests) import it directly,
while the Typer CLI becomes a *thin* wrapper.

The code is **fully compatible with macOS-MPS**: if an Apple-silicon GPU
is detected we automatically patch ``torch.load`` so that all checkpoints
are remapped onto ``torch.device("mps")``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

__all__ = [
    "batch_synthesize",
    "chunk_text",
    "detect_device",
    "patch_torch_load_for_mps",
    "slugify",
    "synthesize_one",
]

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def detect_device(preferred: str | None = None) -> str:
    """Return the best available device.

    Priority:
    1. *preferred* if the caller explicitly set one.
    2. Apple silicon GPU (``mps``) if available.
    3. CUDA GPU (``cuda``) if available.
    4. CPU.
    """
    if preferred:
        return preferred.lower()
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def patch_torch_load_for_mps() -> None:
    """Monkey-patch ``torch.load`` so checkpoints map to *mps* automatically.

    PyTorch's default *map_location* for ``torch.load`` often points to
    ``cuda``.  On machines without CUDA—i.e. most Macs—this throws an
    error.  We sidestep that by injecting a wrapper that forces
    ``map_location=torch.device("mps")`` when running on Apple silicon.
    """

    if not torch.backends.mps.is_available():
        return  # Nothing to do

    _orig_torch_load = torch.load  # noqa: WPS122

    def _patched_load(*args, **kwargs):  # type: ignore[override]
        if "map_location" not in kwargs:
            kwargs["map_location"] = torch.device("mps")
        return _orig_torch_load(*args, **kwargs)

    torch.load = _patched_load  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def slugify(text: str, max_len: int = 40) -> str:
    """Return a filesystem-safe slug derived from *text*."""
    slug = _SLUG_RE.sub("-", text.strip().lower()).strip("-")
    return slug[:max_len] or "speech"


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split *text* into ~*max_chars* chunks, preserving sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    sentences = _SENTENCE_SPLIT_RE.split(text)
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if buf_len + len(sent) + 1 <= max_chars:
            buf.append(sent)
            buf_len += len(sent) + 1
        else:
            chunks.append(" ".join(buf))
            buf = [sent]
            buf_len = len(sent)
    if buf:
        chunks.append(" ".join(buf))
    return chunks


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


def _lazy_model(device: str) -> ChatterboxTTS:  # noqa: WPS430
    """Cache the model so we only pay start-up cost once."""
    # Use function attributes for a super-simple memoization.
    if not hasattr(_lazy_model, "_cache"):
        _lazy_model._cache = {}  # type: ignore[attr-defined]
    if device not in _lazy_model._cache:  # type: ignore[attr-defined]
        _lazy_model._cache[device] = ChatterboxTTS.from_pretrained(device=device)  # type: ignore[attr-defined]
    return _lazy_model._cache[device]  # type: ignore[attr-defined]


def synthesize_one(
    text: str,
    *,
    output_path: Path,
    audio_prompt_path: Path | None = None,
    device: str | None = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.4,
    max_chars: int = 1200,
    overwrite: bool = False,
) -> None:
    """Synthesize *text* and write a single WAV file to *output_path*."""

    device = detect_device(device)
    patch_torch_load_for_mps()

    if output_path.exists() and not overwrite:
        msg = f"{output_path} exists (pass overwrite=True to replace)"
        raise FileExistsError(msg)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = _lazy_model(device)
    chunks = chunk_text(text, max_chars=max_chars)
    wavs: list[torch.Tensor] = []

    for idx, chunk in enumerate(chunks):
        wav = model.generate(
            chunk,
            audio_prompt_path=str(audio_prompt_path) if idx == 0 and audio_prompt_path else None,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
        wavs.append(wav)

    final_wav = torch.cat(wavs, dim=1) if len(wavs) > 1 else wavs[0]
    ta.save(str(output_path), final_wav, model.sr)


def batch_synthesize(
    inputs: Iterable[tuple[str, str]],  # (content, stem)
    *,
    output_dir: Path,
    device: str | None = None,
    audio_prompt_path: Path | None = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    max_chars: int = 2000,
    overwrite: bool = False,
) -> list[Path]:
    """High-level helper to synthesise multiple entries.

    Parameters
    ----------
    inputs:
        Iterable of ``(text, stem)`` tuples.  The *stem* is used to form the
        filename ``{stem}.wav``.
    output_dir:
        Directory where WAV files will be written.

    Returns
    -------
    list[Path]
        Paths to all generated WAV files.
    """
    output_paths: list[Path] = []
    for text, stem in inputs:
        out_path = output_dir / f"{stem}.wav"
        synthesize_one(
            text,
            output_path=out_path,
            audio_prompt_path=audio_prompt_path,
            device=device,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            max_chars=max_chars,
            overwrite=overwrite,
        )
        output_paths.append(out_path)
    return output_paths
