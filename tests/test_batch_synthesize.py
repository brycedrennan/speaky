import speaky.core as core


def test_batch_synthesize_outputs_mp3(tmp_path, monkeypatch):
    created = []

    def fake_synthesize_one(text, *, output_path, **kwargs):
        output_path.write_text("ok")
        created.append(output_path)

    monkeypatch.setattr(core, "synthesize_one", fake_synthesize_one)

    outputs = core.batch_synthesize([("hi", "foo")], output_dir=tmp_path)

    assert outputs == [tmp_path / "foo.mp3"]
    assert outputs[0].exists()
    assert created[0] == outputs[0]
