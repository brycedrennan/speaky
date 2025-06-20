# Speak

**Text‑to‑Speech made easy with Chatterbox TTS**
Generate natural‑sounding speech from plain text—locally on your GPU/CPU using a single, ergonomic command‑line tool **and** a clean Python API.

---

## Features

* **Voice cloning** from a short audio prompt (optional)
* **Emotion control** via exaggeration & classifier‑free guidance
* **Device auto‑detection** — Apple Silicon (*mps*), CUDA GPUs, or CPU
* **Smart sentence chunking** (NLTK) to handle long passages gracefully
* **Trailing‑silence trimming** so outputs end crisply
* **Glitch / clipping detection** heuristic for cleaner audio
* **Verification via transcription** (Distil‑Whisper) to catch missing words



---

## 🛠 Installation

```bash
uv tool install speak
```

---

## 🚀 Quickstart (CLI)

> The CLI groups everything under a single sub‑command: `speak synth`.

| Task              | Command                                                         |
| ----------------- | --------------------------------------------------------------- |
| Say a sentence    | `speak --text "Hello, world!"`                            |
| Batch from a file | `speak -f script.txt -o voiceovers/`                      |
| Clone a voice     | `speak --text "How do I sound?" --voice my_prompt.wav`    |
| Dial up the drama | `speak --text "This is **exciting**!" --exaggeration 1.2` |

All outputs are WAV files named after the text (or file stem) and saved to the current directory unless you pass `--output-dir`.

### Common flags

* `--cfg-weight FLOAT`  • classifier‑free guidance mix (0‑1)
* `--max-chars INT`  • soft limit per chunk (default 800)
* `--save-chunks`  • keep intermediate WAVs for debugging
* `--overwrite`  • replace existing files

Run `speak synth --help` for the full list.

---

## 🐍 Python API

```python
from pathlib import Path
from speak.core import batch_synthesize

batch_synthesize(
    inputs=[("Hello there!", "greeting")],  # (text, stem)
    output_dir=Path("out"),
)
```

The helper wraps all the goodies—chunking, glitch detection, ASR verification, etc.—while caching the heavy TTS model for speed.

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
- Don’t raise bare `Exception`.

#### Python
- Manage env/deps with **uv** (`uv add|remove`, `uv run -- …`).
- No logging config or side-effects at import time.
- Keep interfaces (CLI, web, etc.) thin; put logic elsewhere.
- Use `typer` for CLI interfaces, `fastapi` for web interfaces, and `pydantic` for data models.
