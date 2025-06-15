"""Core synthesis utilities for the *speak* package.

This revision replaces the previous regex-based sentence splitter with
the SaT model from **wtpsplit**, giving vastly more robust segmentation
across 85 languages.  The default maximum-character threshold for
chunking and synthesis is now **800**.

The rest of the public API is unchanged.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from wtpsplit import SaT  # NEW: state-of-the-art segmentation

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
    """Monkey-patch ``torch.load`` so checkpoints map to *mps* automatically."""
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
# _SENTENCE_SPLIT_RE removed — segmentation now handled by SaT


def slugify(text: str, max_len: int = 40) -> str:
    """Return a filesystem-safe slug derived from *text*."""
    slug = _SLUG_RE.sub("-", text.strip().lower()).strip("-")
    return slug[:max_len] or "speech"


# -- NEW: SaT-powered chunking ------------------------------------------------


def _lazy_segmenter() -> SaT:  # noqa: WPS430
    """Cache the SaT model so we only load it once per session."""
    if not hasattr(_lazy_segmenter, "_cache"):
        # 3-layer “-sm” model: excellent accuracy, tiny footprint
        _lazy_segmenter._cache = SaT("sat-3l-sm")  # type: ignore[attr-defined]
    return _lazy_segmenter._cache  # type: ignore[attr-defined]


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split *text* into ≈*max_chars* chunks, preserving sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    # High-quality multilingual sentence segmentation
    sentences = [s.strip() for s in _lazy_segmenter().split(text)]
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for sent in sentences:
        if not sent:
            continue
        # +1 for the space we add when joining
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
    """Cache the TTS model so we only pay start-up cost once."""
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
    max_chars: int = 800,  # UPDATED default
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
    max_chars: int = 800,  # UPDATED default
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
