from typer.testing import CliRunner

import speaky.cli as cli
from speaky import core


def test_positional_text(monkeypatch, tmp_path):
    called = {}

    def fake_batch(entries, *, output_dir, **kwargs):
        called["entries"] = entries
        called["output_dir"] = output_dir
        return [tmp_path / "x.wav"]

    monkeypatch.setattr(core, "batch_synthesize", fake_batch)

    runner = CliRunner()
    result = runner.invoke(cli.app, ["hello"], catch_exceptions=False)

    assert result.exit_code == 0
    assert called["entries"] == [("hello", "hello")]


def test_builtin_voice(monkeypatch, tmp_path):
    called = {}

    def fake_batch(entries, *, output_dir, audio_prompt_path=None, **kwargs):
        called["audio_prompt_path"] = audio_prompt_path
        return [tmp_path / "x.wav"]

    monkeypatch.setattr("speaky.core.batch_synthesize", fake_batch)
    runner = CliRunner()
    result = runner.invoke(cli.app, ["hello", "-v", "vader"], catch_exceptions=False)

    assert result.exit_code == 0
    from speaky.voices import get_voice_path

    assert called["audio_prompt_path"] == get_voice_path("vader")
