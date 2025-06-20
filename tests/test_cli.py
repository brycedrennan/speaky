from typer.testing import CliRunner

import speaky.cli as cli


def test_positional_text(monkeypatch, tmp_path):
    called = {}

    def fake_batch(entries, *, output_dir, **kwargs):
        called["entries"] = entries
        called["output_dir"] = output_dir
        return [tmp_path / "x.wav"]

    monkeypatch.setattr(cli, "batch_synthesize", fake_batch)
    runner = CliRunner()
    result = runner.invoke(cli.app, ["hello"], catch_exceptions=False)

    assert result.exit_code == 0
    assert called["entries"] == [("hello", "hello")]
