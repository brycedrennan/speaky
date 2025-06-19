from pathlib import Path

import pytest

from speak.glitch_detection import glitchy_tail

DATA_DIR = Path(__file__).parent / "data"


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("noise_01.mp3", True),
        ("noise_02.mp3", True),
        ("noise_03.mp3", True),
        ("noise_04.mp3", True),
        ("speech_01.mp3", False),
    ],
)
def test_glitch_detection_samples(filename: str, expected: bool) -> None:
    """Ensure the heuristic flags noisy/clipped audio and skips clean speech."""
    path = DATA_DIR / filename
    assert glitchy_tail(str(path)) == expected
