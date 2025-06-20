from __future__ import annotations

from . import _suppress_warnings  # noqa: F401 - side effects only
from .transcription import transcribe  # re-export ASR helper

__all__: list[str] = [
    "transcribe",
]
