# Speaky

**Text‑to‑Speech with Voice Cloning made easy with Chatterbox TTS**
Generate natural‑sounding speech from plain text—locally on your GPU/CPU using a single, ergonomic command‑line tool.

<video src="docs/we-hold-these-truths-to-be-self-evident--attenborough.mp4" controls width="300"></video>
<figure>
  <figcaption><strong>Attenborough</strong></figcaption>

  <video src="docs/we-hold-these-truths-to-be-self-evident--attenborough.mp4" controls width="300"></video>
  <audio controls src="docs/we-hold-these-truths-to-be-self-evident--attenborough.mp3"></audio>
</figure>

<figure>
  <figcaption><strong>Vader</strong></figcaption>
  <audio controls src="docs/we-hold-these-truths-to-be-self-evident--vader.mp3"></audio>
</figure>

<figure>
  <figcaption><strong>Trump</strong></figcaption>
  <audio controls src="docs/we-hold-these-truths-to-be-self-evident--trump.mp3"></audio>
</figure>

<figure>
  <figcaption><strong>Her (Scarlett)</strong></figcaption>
  <audio controls src="docs/we-hold-these-truths-to-be-self-evident--her.mp3"></audio>
</figure>

<figure>
  <figcaption><strong>Kamala</strong></figcaption>
  <audio controls src="docs/we-hold-these-truths-to-be-self-evident--kamala.mp3"></audio>
</figure>

<figure>
  <figcaption><strong>Siri</strong></figcaption>
  <audio controls src="docs/we-hold-these-truths-to-be-self-evident--siri.mp3"></audio>
</figure>





## Features

* **Voice cloning** from a short audio prompt (optional)
* **Built‑in voices** accessible via `-v NAME` (or `-v ALL` for every voice)
* **Emotion control** via exaggeration & classifier‑free guidance
* **Device auto‑detection** — Apple Silicon (*mps*), CUDA GPUs, or CPU
* **Smart sentence chunking** (NLTK) to handle long passages gracefully
* **Trailing‑silence trimming** so outputs end crisply
* **Glitch / clipping detection** heuristic for cleaner audio
* **Verification via transcription** (Distil‑Whisper) to catch missing words



---

## Quick Start

```bash
uv tool install speaky
```

---

## 🚀 Quickstart (CLI)

> The CLI exposes a single command: `speak`.

| Task              | Command                                                         |
| ----------------- | --------------------------------------------------------------- |
| Say a sentence    | `speak "Hello, world!"`                            |
| Batch from a file | `speak -f script.txt -o voiceovers/`                      |
| Clone a voice     | `speak "How do I sound?" --voice my_prompt.wav`    |
| Use built-in voice| `speak "Join me" -v vader`                         |
| All built-in voices| `speak "Join me" -v ALL`                         |
| Dial up the drama | `speak "This is **exciting**!" --exaggeration 1.2` |

All outputs are MP3 files named after the text (or file stem) and saved to the current directory unless you pass `--output-dir`.

### Common flags

* `--cfg-weight FLOAT`  • classifier‑free guidance mix (0‑1)
* `--max-chars INT`  • soft limit per chunk (450)
* `--save-chunks`  • keep intermediate WAVs for debugging
* `--overwrite`  • replace existing files

Run `speak --help` for the full list.

---

## 🐍 Python API

```python
from pathlib import Path
from speaky.core import batch_synthesize

batch_synthesize(
    inputs=[("Hello there!", "greeting")],  # (text, stem)
    output_dir=Path("out"),
)
```

The helper wraps all the goodies—chunking, glitch detection, transcription verification, etc.—while caching the heavy TTS model for speed.

## Credit

This project is built on top of [chatterbox open-source TTS](https://github.com/resemble-ai/chatterbox).

## Development

### Quick Commands
 - `make init` create the environment and install dependencies
 - `make help` see available commands
 - `make af` format code
 - `make lint` run linter
 - `make typecheck` run type checker
 - `make test` run tests
 - `make check` run all checks (format, lint, typecheck, test)
 - `uv add pkg` add a python dependency
 - `uv run -- python foo/bar.py` run arbitrary command in python env

### Code Conventions

- Always run `make checku` after making changes.

#### Testing
- Use **pytest** (no test classes).
- Always set `match=` in `pytest.raises`.
- Prefer `monkeypatch` over other mocks.
- Mirror the source-tree layout in `tests/`.

#### Exceptions
- Catch only specific exceptions—never blanket `except:` blocks.
- Don't raise bare `Exception`.

#### Python
- Manage env/deps with **uv** (`uv add|remove`, `uv run -- ...`).
- No logging config or side-effects at import time.
- Keep interfaces (CLI, web, etc.) thin; put logic elsewhere.
- Use `typer` for CLI interfaces, `fastapi` for web interfaces, and `pydantic` for data models.

