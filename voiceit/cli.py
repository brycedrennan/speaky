"""Typer wrapper around the *voiceit.core* utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from tqdm.auto import tqdm

from voiceit.core import batch_synthesize, slugify

app = typer.Typer(add_completion=False, help="VoiceIt — TTS made easy with Chatterbox")


@app.command("synth")
def synthesize(
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
        help="Directory where WAV files are saved.",
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
        help="Path to an audio prompt for voice cloning (optional).",
    ),
    exaggeration: float = typer.Option(
        0.5,
        min=0.0,
        max=2.0,
        help="Emotion intensity/exaggeration.",
        show_default=True,
    ),
    cfg_weight: float = typer.Option(
        0.4,
        min=0.0,
        max=1.0,
        help="Classifier-free guidance weight.",
        show_default=True,
    ),
    max_chars: int = typer.Option(
        1200,
        min=200,
        help="Maximum characters per chunk before the text is split automatically.",
        show_default=True,
    ),
):
    """Entry-point for the *voiceit* executable."""

    if not text and not file:
        typer.secho("Error: provide --text and/or --file/-f", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    # Gather inputs
    entries: list[tuple[str, str]] = []  # (text, stem)
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

    # Synthesize (progress bar if >1)
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
        )

    typer.secho("Done!", fg=typer.colors.GREEN)


# Allow `python -m voiceit.cli`


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
