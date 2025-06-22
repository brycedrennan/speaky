# Speaky

**Text‑to‑Speech with Voice Cloning made easy with Chatterbox TTS**

Generate natural‑sounding speech from plain text—locally on your GPU/CPU using a single, ergonomic command‑line tool.

## Features

* **Voice cloning** from a short audio prompt (optional)
* **Built‑in voices** accessible via `-v NAME` (or `-v ALL` for every voice)
* **Emotion control** via exaggeration & classifier‑free guidance
* **Device auto‑detection** — Apple Silicon (*mps*), CUDA GPUs, or CPU
* **Smart sentence chunking** (NLTK) to handle long passages gracefully
* **Trailing‑silence trimming** so outputs end crisply
* **Glitch / clipping detection** heuristic for cleaner audio
* **Verification via transcription** (Distil‑Whisper) to catch missing words

## 🚀 Quickstart (CLI)
Run without installation:
```bash
uvx speak "Do or do not. There is no try."  --voice vader
```

or install first:
```bash
uv tool install speaky

speak --file speech.txt  --voice attenborough
```

Run `speak --help` for the full list of options.

## 🐍 Python API

```python
from pathlib import Path
from speaky.core import batch_synthesize

batch_synthesize(
    inputs=[("Hello there!", "greeting")],  # (text, stem)
    output_dir=Path("out"),
)
```

## Examples
https://github.com/user-attachments/assets/f1956d1a-c407-4944-b9f8-a5402f71cbd8

https://github.com/user-attachments/assets/8e6a81df-773c-4d48-876b-4cced0b4f643

https://github.com/user-attachments/assets/2c2884e2-759a-4b1d-884c-c1d5eed9ddf0

https://github.com/user-attachments/assets/2d70a497-b156-40d8-907f-0beb33ea68d6

https://github.com/user-attachments/assets/0ab6195d-ae9e-420e-bde2-719dad772563

https://github.com/user-attachments/assets/aeabd1d9-3839-4190-8460-8ad2e364dd02

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
