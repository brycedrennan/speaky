# File: /Users/bryce/projects/speak/speak/cli.py
"""Typer wrapper around the *speak.core* utilities."""

from __future__ import annotations

import base64
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:  # pragma: no cover - hints only
    from tqdm.auto import tqdm as _tqdm  # noqa: F401

    from speaky.core import (
        batch_synthesize as _batch_synthesize,  # noqa: F401
        slugify as _slugify,  # noqa: F401
    )

app = typer.Typer(add_completion=False, help="Speak — TTS made easy with Chatterbox")


@app.command(name=None)
def synthesize(
    text_args: list[str] = typer.Argument(
        None,
        metavar="TEXT",
        help="Text to synthesise (can be given without --text).",
    ),
    # Input
    text: str | None = typer.Option(
        None,
        "--text",
        metavar="TEXT",
        help="Text to synthesise (mutually inclusive with --file).",
    ),
    file: list[Path] = typer.Option(
        None,
        "--file",
        "-f",
        help="Path to a UTF-8 text file. Can be given multiple times.",
    ),
    # Output
    output_dir: Path = typer.Option(
        Path("."),
        "--output-dir",
        "-o",
        help="Directory where MP3 files are saved (ignored when --remote is set).",
        show_default=True,
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite/--no-overwrite",
        help="Overwrite existing files if they exist.",
    ),
    # Chatterbox options
    device: str = typer.Option(
        None,
        "--device",
        help="Computation device (cuda, mps, cpu). Auto-detect by default.",
    ),
    audio_prompt_path: Path | None = typer.Option(
        None,
        "--voice",
        "-v",
        help="Path to an audio prompt or name of a built-in voice.",
    ),
    exaggeration: float = typer.Option(
        0.6,
        min=0.0,
        max=2.0,
        help="Emotion intensity/exaggeration.",
        show_default=True,
    ),
    cfg_weight: float = typer.Option(
        0.5,
        min=0.0,
        max=1.0,
        help="Classifier-free guidance weight.",
        show_default=True,
    ),
    max_chars: int = typer.Option(
        450,
        min=200,
        help="Maximum characters per chunk before the text is split automatically.",
        show_default=True,
    ),
    # Debugging / inspection
    save_chunks: bool = typer.Option(
        False,
        "--save-chunks/--no-save-chunks",
        help="Write each generated chunk to a 'speak-chunks' folder alongside the final MP3. Useful for debugging.",
    ),
):
    """Entry-point for the *speak* executable."""
    from tqdm.auto import tqdm

    from speaky.core import batch_synthesize, slugify
    from speaky.voices import available_voices, get_voice_path

    if not text and text_args:
        text = " ".join(text_args)

    if not text and not file:
        typer.secho("Error: provide TEXT and/or --file/-f", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    if audio_prompt_path and not audio_prompt_path.exists():
        candidate = audio_prompt_path.stem
        try:
            audio_prompt_path = get_voice_path(candidate)
        except KeyError:
            available = ", ".join(sorted(available_voices()))
            typer.secho(
                f"Unknown voice '{candidate}'. Available voices: {available}",
                fg=typer.colors.RED,
                err=True,
            )
            raise typer.Exit(code=1)

    # ---------------------------------------------------------------------
    # Gather inputs
    # ---------------------------------------------------------------------
    entries: list[tuple[str, str]] = []  # (text, stem)
    remote = False
    if text:
        entries.append((text, slugify(text)))
    for path in file or []:
        try:
            content = path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError as exc:
            typer.secho(f"Error reading {path}: {exc}", fg=typer.colors.RED, err=True)
            raise typer.Exit(code=1)
        if not content:
            typer.secho(f"Warning: {path} is empty — skipping.", fg=typer.colors.YELLOW, err=True)
            continue
        entries.append((content, path.stem))

    if not entries:
        typer.secho("No valid input found.", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # ---------------------------------------------------------------------
    # Choose execution mode
    # ---------------------------------------------------------------------
    if remote:
        # --------------------------------------------------------------
        # Remote (Modal) execution
        # --------------------------------------------------------------

        import modal

        from speaky import remote_modal

        prompt_bytes = audio_prompt_path.read_bytes() if audio_prompt_path else None

        typer.secho("Submitting job to Modal…", fg=typer.colors.BLUE)
        with modal.enable_output(), remote_modal.app.run():  # Ephemeral app
            results = remote_modal.tts_remote.remote(
                entries=entries,
                device=device,
                audio_prompt_bytes=prompt_bytes,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                max_chars=max_chars,
            )

        # Write returned audio to local disk
        out_dir = output_dir or Path(".")
        out_dir.mkdir(parents=True, exist_ok=True)
        for fname, b64_audio in results:
            (out_dir / fname).write_bytes(base64.b64decode(b64_audio))
            typer.secho(f"Saved {fname}", fg=typer.colors.GREEN)
        return

    # --------------------------------------------------------------
    # Local execution (default)
    # --------------------------------------------------------------
    total = len(entries)
    iter_entries = tqdm(entries, desc="Synthesising", unit="file", colour="green") if total > 1 else entries

    for text_entry, stem in iter_entries:
        batch_synthesize(
            [(text_entry, stem)],
            output_dir=output_dir,
            device=device,
            audio_prompt_path=audio_prompt_path,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            max_chars=max_chars,
            overwrite=overwrite,
            save_chunks=save_chunks,
        )

    typer.secho("Done!", fg=typer.colors.GREEN)


# Allow `python -m speak.cli`
def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
