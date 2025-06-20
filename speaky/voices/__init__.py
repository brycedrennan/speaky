from __future__ import annotations

from pathlib import Path

VOICE_DIR = Path(__file__).parent

__all__ = ["available_voices", "get_voice_path"]


def available_voices() -> set[str]:
    """Return a set of available built-in voice names."""
    return {p.stem for p in VOICE_DIR.glob("*.mp3")}


def get_voice_path(name: str) -> Path:
    """Return the path to the built-in voice *name*.

    Raises
    ------
    KeyError
        If no built-in voice matches *name*.
    """
    candidate = VOICE_DIR / f"{name}.mp3"
    if not candidate.is_file():
        raise KeyError(name)
    return candidate
