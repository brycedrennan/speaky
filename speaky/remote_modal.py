"""Modal app for remote TTS synthesis, rebuilt to use *uv* just like **logojudge**.

This mirrors the ultra-fast build pattern from *logojudge/backend/deploy/modal_app.py*:
we copy the `uv` binary, then run `uv sync` against **pyproject.toml**/`uv.lock`
instead of relying on the (now-discouraged) ``Image.pip_install`` helper.

The function ``tts_remote`` is invoked by *speak.cli* when users call
`speak synth --remote`, and returns a list ``[(filename, base64_mp3_bytes), …]``.
"""

from __future__ import annotations

import base64
import os
from functools import lru_cache
from pathlib import Path

import modal
import pathspec

from speaky.core import batch_synthesize

# ------------------------------------------------------------------------------
#   Project-scoped helpers for selecting which files are sent to Modal
# ------------------------------------------------------------------------------

_SENTINELS: set[str] = {".git", ".hg", "pyproject.toml"}

_ALWAYS_IGNORE: list[str] = [
    ".git",
    ".hg",
    "__pycache__",
    ".direnv",
    ".eggs",
    ".mypy_cache",
    ".nox",
    ".tox",
    ".venv",
    "venv",
    ".svn",
    ".ipynb_checkpoints",
    "_build",
    "buck-out",
    "build",
    "dist",
    "__pypackages__",
]

# We skip resource-heavy binary file types when sending code to Modal
_EXCLUDE_EXTS: tuple[str, ...] = (
    ".mp3",
    ".wav",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".mp4",
)


def _find_project_root(start: str | Path = ".") -> Path | None:
    """Return the first directory up from *start* that contains any *_SENTINELS*."""
    path = Path(start).resolve()
    for parent in [path, *path.parents]:
        if any((parent / s).exists() for s in _SENTINELS):
            return parent
    return None


def _gitignore_spec(root: Path) -> pathspec.PathSpec:
    """Compile ``pathspec.PathSpec`` from .gitignore + _ALWAYS_IGNORE."""
    patterns: list[str] = _ALWAYS_IGNORE.copy()
    gi = root / ".gitignore"
    if gi.is_file():
        patterns.extend(gi.read_text(encoding="utf-8").splitlines())
    return pathspec.PathSpec.from_lines("gitwildmatch", patterns)


def _iter_files(
    directory: Path,
    ignore: pathspec.PathSpec,
    *,
    exclude_exts: tuple[str, ...] = (),
) -> list[Path]:
    """Return every file under *directory* that is **not** ignored."""
    selected: list[Path] = []

    for root, dirs, files in os.walk(directory):
        # Prune ignored directories **in-place** for efficiency
        dirs[:] = [d for d in dirs if not ignore.match_file(os.path.relpath(Path(root, d), directory))]

        for name in files:
            rel = os.path.relpath(Path(root, name), directory)
            if ignore.match_file(rel):
                continue
            if exclude_exts and Path(name).suffix in exclude_exts:
                continue
            selected.append(Path(root, name))

    selected.sort()
    return selected


@lru_cache(maxsize=1)
def _files_for_inclusion(start: str | Path = ".") -> set[Path]:
    """Return a cached *set* of files that will be copied into the Modal image."""
    root = _find_project_root(start)
    if root is None:
        return set()
    spec = _gitignore_spec(root)
    return set(_iter_files(root, spec, exclude_exts=_EXCLUDE_EXTS))


def _file_filter(path: str | Path) -> bool:
    """Return *True* iff *path* should be **excluded** from the Modal build context."""
    return Path(path).absolute() not in _files_for_inclusion()


# ------------------------------------------------------------------------------
#   Modal image & app
# ------------------------------------------------------------------------------

PROJECT_SLUG = "speak"

app = modal.App(name=PROJECT_SLUG)

# (Optionally) reference shared secrets here if you have any
secrets: list[modal.Secret] = []

image = (
    modal.Image.debian_slim()
    # Ultra-fast, “uv”-driven build — mirrors logojudge
    .dockerfile_commands(
        [
            # 1) Prepare environment variables for uv`s multi-layer caching
            ("ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=0 UV_PROJECT_ENVIRONMENT=/usr/local"),
            # 2) Copy the uv binary into the image
            "COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/",
            # 3) Ship *just* our dependency manifests for dependency resolution
            "COPY uv.lock pyproject.toml ./",
            # 4) Resolve + install ALL dependencies (no project code yet!)
            "RUN uv sync --frozen --all-extras --no-install-project",
        ]
    )
    # 5) Finally, copy *only* the project source tree (respecting .gitignore)
    .add_local_dir(".", remote_path="/root", ignore=_file_filter)
)

# ------------------------------------------------------------------------------
#   Remote entry-point for TTS synthesis
# ------------------------------------------------------------------------------


@app.function(
    image=image,
    secrets=secrets,
    timeout=60 * 20,  # generous 20-minute cap for large batches
    gpu="any",  # let Modal schedule on GPU if available
)
def tts_remote(
    entries: list[tuple[str, str]],
    *,
    device: str | None = None,
    audio_prompt_bytes: bytes | None = None,
    exaggeration: float = 0.6,
    cfg_weight: float = 0.5,
    max_chars: int = 800,
) -> list[tuple[str, str]]:
    """Synthesise *entries* and return ``[(filename, base64_audio), …]`` ready
    for download by the caller.

    Parameters
    ----------
    entries
        ``[(text, stem), …]`` pairs coming from *speak.cli*.
    audio_prompt_bytes
        Optional WAV bytes used for voice cloning (sent **once**).
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        prompt_path: Path | None = None

        if audio_prompt_bytes:
            prompt_path = tmpdir_path / "prompt.wav"
            prompt_path.write_bytes(audio_prompt_bytes)

        # Directory for the mp3 outputs
        out_dir = tmpdir_path / "out"
        out_dir.mkdir(exist_ok=True)

        # --------------------------------------------------------------------------------
        # Run synthesis
        # --------------------------------------------------------------------------------
        batch_synthesize(
            inputs=entries,
            output_dir=out_dir,
            device=device,
            audio_prompt_path=prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            max_chars=max_chars,
            overwrite=True,
        )

        # --------------------------------------------------------------------------------
        # Package results as base-64 so we can shuttle bytes back to the client
        # --------------------------------------------------------------------------------
        results: list[tuple[str, str]] = []
        for _, stem in entries:
            fname = f"{stem}.mp3"
            audio_bytes = (out_dir / fname).read_bytes()
            b64_audio = base64.b64encode(audio_bytes).decode()
            results.append((fname, b64_audio))

        return results
