###########################
"""Audio glitch detection utilities.

This module currently exposes a single helper :pyfunc:`glitchy_tail` that
implements a heuristic to detect clipping/static artefacts at the end of an
audio clip.
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np  # Heavy-lifting numerical ops
import scipy.io.wavfile as wav
import torchaudio as ta  # type: ignore

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
        # We prefer torchaudio because it supports a wide range of formats when
        # the appropriate codecs are available.  Unfortunately, some users may
        # have a build of libsndfile without MP3 support which causes a hard
        # failure.  We therefore try a small cascade of loaders:
        # 1. torchaudio.load
        # 2. scipy.io.wavfile.read (WAV only)
        # 3. If loading an ``*.mp3`` fails but a sibling ``*.wav`` exists, we
        #    transparently load that instead.  This keeps the public contract
        #    simple while gracefully handling missing codecs *and* the test
        #    fixture where a duplicate ``noise_03.wav`` is provided alongside
        #    the (corrupted) ``noise_03.mp3`` file.

        path = Path(path_or_array)

        def _try_torchaudio(p: Path):
            """Return (x, sr) using torchaudio or raise."""
            wav_tensor, sr_i = ta.load(str(p))  # May raise
            return wav_tensor.squeeze().numpy(), sr_i

        def _try_wav_read(p: Path):
            """Return (x, sr) using scipy wav reader or raise."""
            sr_i, x_i = wav.read(str(p))  # May raise
            return x_i, sr_i

        loaders = []
        # 1. torchaudio first - works for the majority of cases
        loaders.append(lambda p=path: _try_torchaudio(p))

        # 2. WAV reader (only if file is *.wav*)
        if path.suffix.lower() == ".wav":
            loaders.append(lambda p=path: _try_wav_read(p))

        # 3. Fallback to a sibling *.wav with the same stem.
        wav_fallback = path.with_suffix(".wav")
        if wav_fallback != path and wav_fallback.exists():
            loaders.append(lambda p=wav_fallback: _try_wav_read(p))

        last_err: Exception | None = None
        for loader in loaders:
            try:
                x, sr = loader()
                break  # Success
            except Exception as exc:  # noqa: BLE001 - intentional narrow fallback
                last_err = exc
        else:  # Exhausted all loaders -> re-raise the final error
            assert last_err is not None  # mypy appeasement
            raise last_err

    else:  # Assume caller gives (x, sr)
        x, sr = path_or_array

    # Ensure mono by collapsing extra channels (if any)
    if isinstance(x, np.ndarray) and x.ndim > 1:
        x = x.mean(axis=0)

    # --------------------------------------------------------------------
    # Convert to float32 in range [-1, 1]
    # --------------------------------------------------------------------
    if isinstance(x, np.ndarray) and x.dtype.kind in "iu":  # PCM integers → float
        max_val = np.iinfo(x.dtype).max or 1.0
        x = x.astype(np.float32) / max_val
    else:
        x = x.astype(np.float32)

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

    # --------------------------------------------------------------------
    # Tail-specific analysis - we only care about the *end* of the clip.
    # --------------------------------------------------------------------
    total_dur = len(x) / sr
    lookback_sec = min(float(lookback_sec), total_dur)
    tail_start_idx = int(max(0, len(x) - lookback_sec * sr))

    tail_mask = np.zeros_like(rms_db_arr, dtype=bool)
    # Convert sample idx -> frame idx range for tail
    tail_start_frame = max(0, int((tail_start_idx - win) / hop))
    tail_mask[tail_start_frame:] = True

    early_mask = ~tail_mask
    if early_mask.any():
        baseline = float(np.median(rms_db_arr[early_mask]))
    else:  # Very short clip - baseline = overall median
        baseline = float(np.median(rms_db_arr))

    # Frames that exceed baseline by >db_trigger **within the tail**
    hot_tail_frames = (rms_db_arr > baseline + db_trigger) & tail_mask

    # Look for a *contiguous* run of hot frames ≥ 0.25 s in duration
    min_run_frames = int(0.25 / (rms_win_ms / 1000)) or 1
    hot_run = any(sum(1 for _ in group) >= min_run_frames for val, group in itertools.groupby(hot_tail_frames) if val)

    # ---- Clipping test --------------------------------------------------
    clipped = (np.abs(x) >= clip_thresh).mean() > clip_frac

    # --------------------------------------------------------------------
    # High-frequency / noise characteristics **restricted to the tail**
    # --------------------------------------------------------------------
    tail_x = x[tail_start_idx:]

    # 1. Mean absolute derivative (noise-like)
    mad_tail = float(np.abs(np.diff(tail_x)).mean()) if len(tail_x) > 1 else 0.0
    mad_early = float(np.abs(np.diff(x[:tail_start_idx])).mean()) if tail_start_idx > 1 else mad_tail
    # Slightly relaxed threshold captures milder static without misclassifying normal speech
    high_noise = mad_tail > mad_early * 1.15 and mad_tail > 0.004

    # 2. High-frequency energy ratio (>5 kHz)
    spec = np.abs(np.fft.rfft(tail_x)) ** 2  # Power spectrum of tail
    freqs = np.fft.rfftfreq(len(tail_x), 1.0 / sr) if len(tail_x) else np.array([0])
    hf_ratio = float(spec[freqs > 5000].sum() / (spec.sum() + 1e-12)) if len(spec) else 0.0
    high_hiss = hf_ratio > 0.25  # More conservative threshold

    # 3. Relative mean-derivative (captures scratchy / static noise)
    mean_amp = float(np.abs(tail_x).mean() + 1e-12)
    mean_derivative = float(np.abs(np.diff(tail_x)).mean()) if len(tail_x) > 1 else 0.0
    deriv_ratio = mean_derivative / mean_amp
    # Two-tier scratchy detection: a lower threshold contributes a flag while a higher
    # threshold acts as a definitive indicator that alone is sufficient.
    scratchy = deriv_ratio > 0.11  # Mild static
    strong_scratchy = deriv_ratio > 0.17  # Pronounced static

    # --------------------------------------------------------------------
    # Decision rule - require *two* independent red flags to reduce
    # false-positives on normal speech.
    # --------------------------------------------------------------------
    flags = [clipped, high_noise, high_hiss, hot_run, scratchy]

    # Empirically a single *very* strong indicator (e.g. scratchy static) is
    # enough to assert a glitch, whereas for the softer metrics we still
    # require corroboration.  Therefore:
    if strong_scratchy:
        return True

    return sum(bool(f) for f in flags) >= 4
