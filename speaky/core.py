# File: /Users/bryce/projects/speak/speak/core.py
"""Core synthesis utilities for the *speak* package."""

from __future__ import annotations

import logging
import re
from functools import cache
from typing import TYPE_CHECKING, cast

import nltk
import torch
import torchaudio as ta

# Local package helpers
from speaky.glitch_detection import glitchy_tail
from speaky.transcription import _chunk_passes_asr
from speaky.tts import ParallelChatterboxTTS, auto_batch_size

if TYPE_CHECKING:  # pragma: no cover
    from collections.abc import Iterable
    from pathlib import Path

__all__ = [
    "batch_synthesize",
    "chunk_text",
    "detect_device",
    "glitchy_tail",  # NEW: export helper
    "patch_torch_load_for_mps",
    "slugify",
    "synthesize_one",
    "trim_trailing_silence",  # NEW: export helper
]

# ---------------------------------------------------------------------------
# Device helpers
# ---------------------------------------------------------------------------


def detect_device(preferred: str | None = None) -> str:
    """Return the best available device.

    Priority:
    1. *preferred* if supplied by the caller.
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

    _orig_load = torch.load  # noqa: WPS122

    def _patched_load(*args, **kwargs):  # type: ignore[override]
        if "map_location" not in kwargs:
            kwargs["map_location"] = torch.device("mps")
        return _orig_load(*args, **kwargs)

    torch.load = _patched_load  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")


def slugify(text: str, max_len: int = 40) -> str:
    """Return a filesystem-safe slug derived from *text*."""
    slug = _SLUG_RE.sub("-", text.strip().lower()).strip("-")
    return slug[:max_len] or "speech"


# -- Sentence chunking with NLTK -------------------------------------------


def _ensure_punkt() -> None:
    """Ensure that the Punkt tokenizer is available."""
    try:
        nltk.data.find("tokenizers/punkt")
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:  # pragma: no cover
        nltk.download(["punkt", "punkt_tab"], quiet=True)


def _sentences(text: str) -> list[str]:
    """Segment *text* into sentences using NLTK."""
    _ensure_punkt()
    return [s.strip() for s in nltk.tokenize.sent_tokenize(text) if s.strip()]


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split *text* into ≈*max_chars* chunks, preserving sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    sentences = _sentences(text)
    chunks: list[str] = []
    buf: list[str] = []
    buf_len = 0

    for sent in sentences:
        # +1 accounts for the space we add when joining
        candidate_len = buf_len + len(sent) + (1 if buf else 0)
        if candidate_len <= max_chars:
            buf.append(sent)
            buf_len = candidate_len
        else:
            chunks.append(" ".join(buf))
            buf = [sent]
            buf_len = len(sent)
    if buf:
        chunks.append(" ".join(buf))
    chunks = [c.strip() for c in chunks if c.strip()]
    return chunks


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


@cache
def _lazy_model(device: str) -> ParallelChatterboxTTS:  # noqa: WPS430
    """Cache the TTS model so we only pay start-up cost once."""
    return cast("ParallelChatterboxTTS", ParallelChatterboxTTS.from_pretrained(device=device))


def synthesize_one(
    text: str,
    *,
    output_path: Path,
    audio_prompt_path: Path | None = None,
    device: str | None = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.4,
    max_chars: int = 800,
    overwrite: bool = False,
    save_chunks: bool = False,
    save_rejects: bool = False,
    min_chunk_seconds: float = 0.3,
    min_sec_per_word: float = 0.12,
    max_retries: int = 3,
    max_trailing_silence: float = 0.7,
    verify_with_asr: bool = True,
    asr_model_size: str = "large",
    max_missing_ratio: float = 0.02,
) -> None:
    """Synthesize *text* and write a single MP3 file to *output_path*.

    When *verify_with_asr* is set, each generated chunk is fed through an
    (optional) *faster-whisper* ASR model. The chunk is accepted only if the
    majority of words (as defined by *max_missing_ratio*) appear in the
    transcription, providing an automatic safeguard that the TTS actually
    uttered the requested text.

    If ``save_chunks`` is ``True`` the final audio for each chunk is written to
    a ``speak-chunks`` directory.  When ``save_rejects`` is also ``True`` all
    failed attempts are saved to ``speak-rejects`` for debugging.
    """
    device = detect_device(device)
    patch_torch_load_for_mps()

    if output_path.exists() and not overwrite:
        msg = f"{output_path} exists (pass overwrite=True to replace)"
        raise FileExistsError(msg)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = _lazy_model(device)
    chunks = chunk_text(text, max_chars=max_chars)
    batch_size = auto_batch_size(device)
    wavs: list[torch.Tensor] = []

    # Track character offsets so we can embed them in filenames
    start_positions: list[int] = []
    search_pos = 0
    if save_chunks:
        chunk_dir = (output_path.parent / "speak-chunks").resolve()
        chunk_dir.mkdir(parents=True, exist_ok=True)
    if save_rejects:
        reject_dir = (output_path.parent / "speak-rejects").resolve()
        reject_dir.mkdir(parents=True, exist_ok=True)
    audio_slug = output_path.stem

    logger = logging.getLogger(__name__)

    for chunk in chunks:
        start_idx = text.find(chunk, search_pos)
        if start_idx == -1:
            start_idx = search_pos
        start_positions.append(start_idx)
        search_pos = start_idx + len(chunk)

    if audio_prompt_path:
        model.prepare_conditionals(str(audio_prompt_path), exaggeration=exaggeration)

    for bstart in range(0, len(chunks), batch_size):
        batch = chunks[bstart : bstart + batch_size]
        batch_wavs = model.generate_batch(
            batch,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
        )
        for j, (chunk, start_idx, generated) in enumerate(
            zip(batch, start_positions[bstart : bstart + batch_size], batch_wavs),
            start=bstart + 1,
        ):
            idx = j
            attempt = 0
            while True:
                # NEW: glitch detection - run BEFORE silence trimming
                raw_np = generated.detach().cpu().numpy().reshape(-1)
                is_glitchy = glitchy_tail((raw_np, model.sr))
                is_glitchy = False

                # Now trim excessive trailing silence for final acceptance
                wav = trim_trailing_silence(generated, model.sr, max_silence_sec=max_trailing_silence)

                duration = wav.shape[-1] / model.sr
                words = len(chunk.split()) or 1
                dynamic_min = max(min_chunk_seconds, words * min_sec_per_word)

                if duration < dynamic_min:
                    logger.warning(
                        "Chunk %s attempt %s flagged as too short (%.2fs < %.2fs). Retrying…",
                        idx,
                        attempt,
                        duration,
                        dynamic_min,
                    )

                if is_glitchy:
                    logger.warning(
                        "Chunk %s attempt %s appears glitchy/clipped according to heuristic. Retrying…",
                        idx,
                        attempt,
                    )

                asr_ok = True
                if verify_with_asr:
                    try:
                        asr_ok = _chunk_passes_asr(
                            wav,
                            model.sr,
                            chunk,
                            device,
                            max_missing_ratio=max_missing_ratio,
                            model_size=asr_model_size,
                        )
                    except Exception as exc:  # noqa: BLE001 - any ASR failure triggers retry
                        logger.warning(
                            "Chunk %s attempt %s failed ASR verification due to error: %s. Retrying…",
                            idx,
                            attempt,
                            exc,
                        )
                        asr_ok = False

                if verify_with_asr and not asr_ok:
                    logger.warning(
                        "Chunk %s attempt %s failed ASR word match threshold. Retrying…",
                        idx,
                        attempt,
                    )

                meet_quality = duration >= dynamic_min and not is_glitchy and (not verify_with_asr or asr_ok)

                if meet_quality or attempt >= max_retries:
                    break
                if save_rejects:
                    chunk_slug = slugify(chunk, max_len=30)
                    rname = f"{audio_slug}_{idx}_{start_idx}_{chunk_slug}_attempt{attempt}.mp3"
                    ta.save(str(reject_dir / rname), wav, model.sr)
                attempt += 1
                generated = model.generate(
                    chunk,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                )

        # -----------------------------------------------------------------
        # Optionally write chunk WAV to disk for inspection/debugging
        # -----------------------------------------------------------------
        if save_chunks:
            chunk_slug = slugify(chunk, max_len=50)
            fname = f"{audio_slug}_{idx}_{start_idx}_{chunk_slug}.wav"
            ta.save(str(chunk_dir / fname), wav, model.sr)

        wavs.append(wav)

    final_wav = torch.cat(wavs, dim=1) if len(wavs) > 1 else wavs[0]
    ta.save(str(output_path), final_wav, model.sr, format="mp3")


def batch_synthesize(
    inputs: Iterable[tuple[str, str]],
    *,
    output_dir: Path,
    device: str | None = None,
    audio_prompt_path: Path | None = None,
    exaggeration: float = 0.5,
    cfg_weight: float = 0.5,
    max_chars: int = 800,
    overwrite: bool = False,
    save_chunks: bool = False,
    save_rejects: bool = False,
    min_chunk_seconds: float = 0.3,
    min_sec_per_word: float = 0.12,
    max_retries: int = 3,
    max_trailing_silence: float = 0.7,
    verify_with_asr: bool = True,
    asr_model_size: str = "large",
    max_missing_ratio: float = 0.02,
) -> list[Path]:
    """High-level helper to synthesise multiple entries.

    The ``save_chunks`` and ``save_rejects`` flags mirror those in
    :func:`synthesize_one` and control whether intermediate audio files are
    written to disk for debugging.
    """
    output_paths: list[Path] = []
    for text, stem in inputs:
        out_path = output_dir / f"{stem}.mp3"
        synthesize_one(
            text,
            output_path=out_path,
            audio_prompt_path=audio_prompt_path,
            device=device,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            max_chars=max_chars,
            overwrite=overwrite,
            save_chunks=save_chunks,
            save_rejects=save_rejects,
            min_chunk_seconds=min_chunk_seconds,
            min_sec_per_word=min_sec_per_word,
            max_retries=max_retries,
            max_trailing_silence=max_trailing_silence,
            verify_with_asr=verify_with_asr,
            asr_model_size=asr_model_size,
            max_missing_ratio=max_missing_ratio,
        )
        output_paths.append(out_path)
    return output_paths


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def trim_trailing_silence(
    wav: torch.Tensor,
    sr: int,
    max_silence_sec: float = 0.7,
    silence_thresh_db: float = -45.0,
) -> torch.Tensor:
    """Return *wav* with excessive *trailing* silence trimmed.

    Parameters
    ----------
    wav : torch.Tensor
        Audio tensor shaped ``(channels, samples)`` in the range ``[-1, 1]``.
    sr : int
        Sample-rate of *wav*.
    max_silence_sec : float, optional
        Maximum trailing silence to *retain*, by default ``0.7`` seconds.
    silence_thresh_db : float, optional
        Samples whose amplitude is below this dBFS threshold are considered
        "silent", by default ``-45 dB``.

    Returns
    -------
    torch.Tensor
        Trimmed audio tensor (a view when no trimming is necessary).
    """

    if wav.ndim != 2 or wav.shape[-1] == 0:
        return wav  # Unexpected shape or empty - leave untouched

    # Compute amplitude threshold from dB FS value
    amp_thresh = 10 ** (silence_thresh_db / 20.0)

    # Collapse channels by taking the maximum magnitude per sample
    mag = wav.abs().max(dim=0).values

    # Find last non-silent sample index
    non_silence_idx = (mag > amp_thresh).nonzero(as_tuple=False)
    if non_silence_idx.numel() == 0:
        return wav  # All silence - do not risk returning empty tensor

    last_loud = int(non_silence_idx[-1])
    keep_until = min(wav.shape[-1], last_loud + int(max_silence_sec * sr))

    if keep_until < wav.shape[-1]:
        wav = wav[:, :keep_until]

    return wav
