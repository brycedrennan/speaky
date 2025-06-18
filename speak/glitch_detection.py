###########################
"""Audio glitch detection utilities.

This module currently exposes a single helper :pyfunc:`glitchy_tail` that
implements a heuristic to detect clipping/static artefacts at the end of an
audio clip.
"""

from __future__ import annotations

import numpy as np  # Heavy-lifting numerical ops
import scipy.io.wavfile as wav

__all__ = [
    "glitchy_tail",
]


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

    # ---- Load -----------------------------------------------------------
    if isinstance(path_or_array, str):
        sr, x = wav.read(path_or_array)
    else:  # Assume caller gives (x, sr)
        x, sr = path_or_array

    # Convert to float32 in range [-1, 1]
    x = x.astype(np.float32)
    if x.dtype.kind in "iu":  # PCM → float
        x /= np.iinfo(x.dtype).max or 1.0

    # ---- Frame-level RMS (dB) ------------------------------------------
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

    # ---- Split: body vs tail -------------------------------------------
    tail_frames = int(lookback_sec / (rms_win_ms / 1000))
    tail = rms_db_arr[-tail_frames:]
    body = rms_db_arr[:-tail_frames] if len(rms_db_arr) > tail_frames else rms_db_arr

    baseline = np.median(body) if len(body) else np.median(rms_db_arr)
    loud_tail = (tail > baseline + db_trigger).mean() > 0.5  # >50 % hot

    # --------------------------------------------------------------------
    # High-frequency/noise test based on mean absolute derivative
    # --------------------------------------------------------------------
    tail_samples = x[-int(lookback_sec * sr) :] if "x" in locals() else None  # type: ignore[name-defined]
    if tail_samples is None:
        return bool(loud_tail)

    mad_tail = float(np.abs(np.diff(tail_samples)).mean()) if len(tail_samples) > 1 else 0.0
    high_noise = mad_tail > 0.01  # Empirically tuned on sample data

    # --------------------------------------------------------------------
    # High-frequency energy ratio (>5 kHz) — catches quiet static hiss
    # --------------------------------------------------------------------
    spec = np.abs(np.fft.rfft(tail_samples)) ** 2  # Power spectrum
    freqs = np.fft.rfftfreq(len(tail_samples), 1.0 / sr)
    hf_ratio = float(spec[freqs > 5000].sum() / (spec.sum() + 1e-12))
    high_hiss = hf_ratio > 0.1  # Tuned on corpus (good≈0.002, hiss≈0.18)

    return bool(loud_tail or high_noise or high_hiss)
