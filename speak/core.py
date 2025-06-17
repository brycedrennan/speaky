# File: /Users/bryce/projects/speak/speak/core.py
"""Core synthesis utilities for the *speak* package."""

from __future__ import annotations

import re
from functools import cache
from typing import TYPE_CHECKING

import nltk  # Using NLTK for sentence tokenization
import numpy as np  # NEW: for glitch detection
import scipy.io.wavfile as wav  # NEW: for glitch detection
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

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
    except LookupError:  # pragma: no cover
        nltk.download("punkt", quiet=True)


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
    return chunks


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------


@cache
def _lazy_model(device: str) -> ChatterboxTTS:  # noqa: WPS430
    """Cache the TTS model so we only pay start-up cost once."""
    return ChatterboxTTS.from_pretrained(device=device)


def glitchy_tail(
    path_or_array,
    lookback_sec: float = 3.0,
    rms_win_ms: float = 20,
    db_trigger: float = 6,
    clip_thresh: float = 0.20,
    clip_frac: float = 1e-3,
) -> bool:
    """Heuristic to detect the clipping/glitch artefact observed in some outputs.

    Parameters
    ----------
    path_or_array : str | tuple[np.ndarray, int]
        Path to a WAV file **or** a tuple ``(samples, sr)`` where *samples* is
        a float or integer numpy array and *sr* the sample-rate.
    lookback_sec : float, optional
        Amount of audio (from the *end*) to inspect, by default ``3`` seconds.
    rms_win_ms : float, optional
        Size of the short-time RMS window used to compute the energy profile,
        in milliseconds, by default ``20``.
    db_trigger : float, optional
        How many dB the tail may exceed the median RMS of the *earlier* part
        of the file before it is considered a glitch, by default ``6``.
    clip_thresh : float, optional
        Absolute sample value that counts as "clipped", by default ``0.20``.
    clip_frac : float, optional
        Fraction of samples that must exceed *clip_thresh* to flag clipping,
        by default ``1e-3``.

    Returns
    -------
    bool
        ``True`` if the tail appears glitchy / clipped, ``False`` otherwise.
    """
    # ---- Load ----
    if isinstance(path_or_array, str):
        sr, x = wav.read(path_or_array)
    else:  # Assume caller gives (x, sr)
        x, sr = path_or_array

    # Convert to float32 in range [-1, 1]
    x = x.astype(np.float32)
    if x.dtype.kind in "iu":  # PCM → float
        x /= np.iinfo(x.dtype).max or 1.0

    # ---- Frame-level RMS (dB) ----
    hop = win = int(sr * rms_win_ms / 1000)
    if hop == 0:
        return False  # Degenerate
    rms_db: list[float] = []
    for i in range(0, len(x) - win, hop):
        frame = x[i : i + win]
        rms = np.sqrt(np.mean(frame**2) + 1e-12)
        rms_db.append(20 * np.log10(rms + 1e-12))
    if not rms_db:
        return False  # short clip — treat as non-glitchy

    rms_db_arr = np.asarray(rms_db)

    # ---- Split: body vs tail ----
    tail_frames = int(lookback_sec / (rms_win_ms / 1000))
    tail = rms_db_arr[-tail_frames:]
    body = rms_db_arr[:-tail_frames] if len(rms_db_arr) > tail_frames else rms_db_arr

    baseline = np.median(body) if len(body) else np.median(rms_db_arr)
    loud_tail = (tail > baseline + db_trigger).mean() > 0.5  # >50 % hot

    # ------------------------------------------------------------
    # High-frequency/noise test based on mean absolute derivative
    # ------------------------------------------------------------
    # Need waveform samples for derivative; reload if not already present
    tail_samples = x[-int(lookback_sec * sr) :] if "x" in locals() else None  # type: ignore[name-defined]
    if tail_samples is None:
        # Safeguard: if we somehow lost reference, fallback to non-glitch
        return bool(loud_tail)

    mad_tail = float(np.abs(np.diff(tail_samples)).mean()) if len(tail_samples) > 1 else 0.0
    high_noise = mad_tail > 0.01  # Empirically tuned on sample data

    # ------------------------------------------------------------
    # High-frequency energy ratio (>5 kHz) — catches quiet static hiss
    # ------------------------------------------------------------
    spec = np.abs(np.fft.rfft(tail_samples)) ** 2  # Power spectrum
    freqs = np.fft.rfftfreq(len(tail_samples), 1.0 / sr)
    hf_ratio = float(spec[freqs > 5000].sum() / (spec.sum() + 1e-12))
    high_hiss = hf_ratio > 0.1  # Tuned on corpus (good≈0.002, hiss≈0.18)

    return bool(loud_tail or high_noise or high_hiss)


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
    min_chunk_seconds: float = 0.3,
    min_sec_per_word: float = 0.12,
    max_retries: int = 3,
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

    # Track character offsets so we can embed them in filenames
    search_pos = 0
    if save_chunks:
        chunk_dir = (output_path.parent / "speak-chunks").resolve()
        chunk_dir.mkdir(parents=True, exist_ok=True)
    audio_slug = output_path.stem

    for idx, chunk in enumerate(chunks, start=1):
        # -----------------------------------------------------------------
        # Locate starting character index for this chunk (best-effort)
        # -----------------------------------------------------------------
        start_idx = text.find(chunk, search_pos)
        if start_idx == -1:
            start_idx = search_pos  # Fallback
        search_pos = start_idx + len(chunk)

        # -----------------------------------------------------------------
        # Generate audio, retry if duration is suspiciously short OR glitchy
        # -----------------------------------------------------------------
        attempt = 0
        while True:
            wav = model.generate(
                chunk,
                audio_prompt_path=str(audio_prompt_path) if idx == 1 and audio_prompt_path else None,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
            duration = wav.shape[-1] / model.sr
            # Dynamic minimum based on text length (word count)
            words = len(chunk.split()) or 1
            dynamic_min = max(min_chunk_seconds, words * min_sec_per_word)

            # NEW: glitch detection
            raw_np = wav.detach().cpu().numpy().reshape(-1)
            is_glitchy = glitchy_tail((raw_np, model.sr))

            if (duration >= dynamic_min and not is_glitchy) or attempt >= max_retries:
                break
            attempt += 1

        # -----------------------------------------------------------------
        # Optionally write chunk WAV to disk for inspection/debugging
        # -----------------------------------------------------------------
        if save_chunks:
            chunk_slug = slugify(chunk, max_len=50)
            fname = f"{audio_slug}_{idx}_{start_idx}_{chunk_slug}.wav"
            ta.save(str(chunk_dir / fname), wav, model.sr)

        wavs.append(wav)

    final_wav = torch.cat(wavs, dim=1) if len(wavs) > 1 else wavs[0]
    ta.save(str(output_path), final_wav, model.sr)


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
    min_chunk_seconds: float = 0.3,
    min_sec_per_word: float = 0.12,
    max_retries: int = 3,
) -> list[Path]:
    """High-level helper to synthesise multiple entries."""
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
            save_chunks=save_chunks,
            min_chunk_seconds=min_chunk_seconds,
            min_sec_per_word=min_sec_per_word,
            max_retries=max_retries,
        )
        output_paths.append(out_path)
    return output_paths
