# File: /Users/bryce/projects/speak/speak/cli.py
"""Typer wrapper around the *speak.core* utilities."""

from __future__ import annotations

import sys
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="Speak — TTS")


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
        Path("./outputs"),
        "--output-dir",
        "-o",
        help="Directory where MP3 files are saved.",
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
    save_rejects: bool = typer.Option(
        False,
        "--save-rejects/--no-save-rejects",
        help="Also save failed generation attempts to 'speak-rejects'.",
    ),
):
    """Entry-point for the *speak* executable."""
    from tqdm.auto import tqdm

    from speaky import (
        _suppress_warnings,  # noqa: F401 - side effects only
        core,
    )
    from speaky.voices import available_voices, get_voice_path

    if not text and text_args:
        text = " ".join(text_args)

    if not text and not file:
        typer.secho("Error: provide TEXT and/or --file/-f", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1)

    voice_all = False
    if audio_prompt_path and not audio_prompt_path.exists():
        candidate = audio_prompt_path.stem
        if candidate.lower() == "all":
            voice_all = True
        else:
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
    if text:
        entries.append((text, core.slugify(text)))
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

    # --------------------------------------------------------------
    # Local execution (default)
    # --------------------------------------------------------------
    total = len(entries)

    if voice_all:
        for voice_name in sorted(available_voices()):
            voice_path = get_voice_path(voice_name)
            tagged = [(text_entry, f"{stem}-{voice_name}") for text_entry, stem in entries]
            tqdm_desc = f"Synthesising {voice_name}" if total > 1 else None
            iter_entries = tqdm(tagged, desc=tqdm_desc, unit="file", colour="green") if tqdm_desc else tagged
            for text_entry, stem in iter_entries:
                core.batch_synthesize(
                    [(text_entry, stem)],
                    output_dir=output_dir,
                    device=device,
                    audio_prompt_path=voice_path,
                    exaggeration=exaggeration,
                    cfg_weight=cfg_weight,
                    max_chars=max_chars,
                    overwrite=overwrite,
                    save_chunks=save_chunks,
                    save_rejects=save_rejects,
                )
    else:
        iter_entries = tqdm(entries, desc="Synthesising", unit="file", colour="green") if total > 1 else entries
        for text_entry, stem in iter_entries:
            core.batch_synthesize(
                [(text_entry, stem)],
                output_dir=output_dir,
                device=device,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
                max_chars=max_chars,
                overwrite=overwrite,
                save_chunks=save_chunks,
                save_rejects=save_rejects,
            )

    typer.secho("Done!", fg=typer.colors.GREEN)


# Allow `python -m speak.cli`
def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
