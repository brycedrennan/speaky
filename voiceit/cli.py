from __future__ import annotations

"""Simple CLI wrapper around the Chatterbox TTS model.

This script exposes a **voiceit** command-line interface powered by `typer`.
It lets you generate spoken‑word WAV files from either:

* **Inline text** provided via the `--text` option, **or**
* One or more **text‑file paths** supplied with `--file / -f` (each file is synthesised separately).

The CLI now **auto‑chunks long texts** so they fit the model's ~2 k‑token limit, stitching the generated
audio back together transparently.

Examples
--------
Generate a single file from a text string:

```bash
voiceit synth --text "Hello world" -o outputs/
```

Generate multiple audio files from text files:

```bash
voiceit synth -f script1.txt -f script2.txt --output-dir out/
```

Both styles can be combined (all inputs are processed):

```bash
voiceit synth --text "Quick demo" -f story.txt
```

The tool also forwards the **most useful Chatterbox options** so you can control emotion, voice cloning
and performance without touching Python code.
"""

from pathlib import Path
import re
import sys
from typing import Iterable, List, Optional

import torch
import torchaudio as ta
import typer
from chatterbox.tts import ChatterboxTTS

# Progress bar (falls back gracefully in unsupported environments)
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Typer app
# ---------------------------------------------------------------------------

app = typer.Typer(add_completion=False, help="VoiceIt — TTS made easy with Chatterbox")

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-zA-Z0-9]+")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _slugify(text: str, max_len: int = 40) -> str:
    """Return a filesystem‑safe slug derived from *text*."""
    slug = _SLUG_RE.sub("-", text.strip().lower()).strip("-")
    return slug[:max_len] or "speech"


def _chunk_text(text: str, max_chars: int) -> List[str]:
    """Split *text* into chunks no longer than *max_chars* while preserving sentence boundaries."""
    if len(text) <= max_chars:
        return [text]

    sentences = _SENTENCE_SPLIT_RE.split(text)
    chunks: List[str] = []
    buf: List[str] = []
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
# Main command
# ---------------------------------------------------------------------------

@app.command("synth")
def synthesize(
    # --- Input ------------------------------------------------------------------
    text: Optional[str] = typer.Option(
        None,
        "--text",
        metavar="TEXT",
        help="Text to synthesise (mutually inclusive with --file).",
    ),
    file: List[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="Path to a UTF‑8 text file. Can be given multiple times.",
    ),
    # --- Output -----------------------------------------------------------------
    output_dir: Path = typer.Option(
        Path("."),
        "--output-dir",
        "-o",
        help="Directory where WAV files are saved.",
        show_default=True,
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite/--no-overwrite",
        help="Overwrite existing files if they exist.",
    ),
    # --- Chatterbox options ------------------------------------------------------
    device: str = typer.Option(
        "cpu",
        help="Device to run inference on (e.g. 'cuda' or 'cpu').",
        case_sensitive=False,
    ),
    audio_prompt_path: Optional[Path] = typer.Option(
        None,
        "--voice",
        "-v",
        help="Path to an audio prompt for voice cloning (optional).",
    ),
    exaggeration: float = typer.Option(
        0.5,
        min=0.0,
        max=1.0,
        help="Emotion intensity/exaggeration (0–1). Higher = more expressive.",
        show_default=True,
    ),
    cfg_weight: float = typer.Option(
        0.5,
        min=0.0,
        max=1.0,
        help="Classifier‑free guidance weight (0–1). Lower = slower, more precise pacing.",
        show_default=True,
    ),
    # --- Chunking ---------------------------------------------------------------
    max_chars: int = typer.Option(
        2000,
        min=200,
        help="Maximum characters per chunk before the text is split automatically.",
        show_default=True,
    ),
):
    """Synthesize speech from inline text and/or text‑file inputs."""

    # -----------------------------------------------------------------------
    # Validate input
    # -----------------------------------------------------------------------
    if not text and not file:
        typer.secho("Error: You must provide either --text or --file/-f. See --help.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    text_entries: List[tuple[str, str]] = []  # (text, stem)

    # Inline text
    if text:
        text_entries.append((text, _slugify(text)))

    # Text files
    for path in file or []:
        if not path.is_file():
            typer.secho(f"Error: '{path}' is not a file or does not exist.", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            typer.secho(f"Error reading '{path}': {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        if not content.strip():
            typer.secho(f"Warning: '{path}' is empty — skipping.", fg=typer.colors.YELLOW, err=True)
            continue
        text_entries.append((content, path.stem))

    if not text_entries:
        typer.secho("No valid input text found.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # -----------------------------------------------------------------------
    # Prepare output directory
    # -----------------------------------------------------------------------
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        typer.secho(f"Error creating output directory '{output_dir}': {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # -----------------------------------------------------------------------
    # Load model once
    # -----------------------------------------------------------------------
    typer.echo("Loading Chatterbox model …")
    model = ChatterboxTTS.from_pretrained(device=device.lower())

    # -----------------------------------------------------------------------
    # Synthesize each entry (chunked if necessary)
    # -----------------------------------------------------------------------
    for idx, (snippet, stem) in enumerate(text_entries, start=1):
        out_path = output_dir / f"{stem}.wav"
        if out_path.exists() and not overwrite:
            typer.secho(f"Skipped {out_path} (file exists, use --overwrite to replace).", fg=typer.colors.YELLOW)
            continue

        typer.echo(f"[{idx}/{len(text_entries)}] Synthesising → {out_path}")

        chunks = _chunk_text(snippet, max_chars=max_chars)
        wavs: List[torch.Tensor] = []

        # Use a tqdm progress bar when synthesising multiple chunks so users
        # can see progress at a glance. For single-chunk inputs, fall back to
        # the existing plain echo to avoid an unnecessary bar flash.
        if len(chunks) > 1:
            chunk_iter = tqdm(chunks, total=len(chunks), unit="chunk", desc="Chunks", colour="green")
        else:
            chunk_iter = chunks  # type: ignore[assignment]

        for c_idx, chunk in enumerate(chunk_iter, start=1):
            if len(chunks) == 1:
                # Preserve the old message style for single-chunk inputs
                typer.echo(f"  • Chunk {c_idx}/{len(chunks)} (len={len(chunk)}) …")

            wav = model.generate(
                chunk,
                audio_prompt_path=str(audio_prompt_path) if c_idx == 1 and audio_prompt_path else None,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
            wavs.append(wav)

        # Concatenate chunks along time dimension (dim=1) and save
        final_wav = torch.cat(wavs, dim=1) if len(wavs) > 1 else wavs[0]
        ta.save(str(out_path), final_wav, model.sr)

    typer.secho("Done!", fg=typer.colors.GREEN)


# ---------------------------------------------------------------------------
# Entrypoint helper
# ---------------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    """Executable entrypoint for `python -m voiceit.cli`."""
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
