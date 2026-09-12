import json
import sys

import pytest
import torch
from aurora_visual.evaluation.metrics import extraction_metrics, grouped_bootstrap, iou
from aurora_visual.evaluation.runner import evaluate, modality_probe
from aurora_visual.training.data import load_manifest, smoke_samples
from aurora_visual.training.engine import load_checkpoint, train


def test_train_save_resume_predict_and_gate(tmp_path):
    samples = smoke_samples()
    path = tmp_path / "model.pt"
    meta, history = train(samples, path, {"epochs": 2})
    model, loaded, _ = load_checkpoint(path)
    assert meta == loaded and history[-1]["train_loss"] < history[0]["train_loss"]
    assert all(torch.isfinite(p).all() for p in model.parameters())
    with pytest.raises(ValueError, match="FIXTURE_ONLY"):
        load_checkpoint(path, for_live=True)
    result = evaluate(model, samples, interventions=False)
    assert result["data_kind"] == "fixture" and not result["research_evaluated"]
    with pytest.raises(ValueError, match="Fixtures"):
        evaluate(model, samples, confirmatory=True)
    meta2, _ = train(samples, path, {"epochs": 3}, resume=path)
    assert meta2["trained_steps"] == 15
    model2, _, _ = load_checkpoint(path)
    assert torch.isfinite(model2(**samples[0]["inputs"])["logits"]).all()
    with pytest.raises(ValueError, match="mismatch"):
        train(smoke_samples(32), path, {"epochs": 4}, resume=path)


def test_checkpoint_tamper(tmp_path):
    path = tmp_path / "a.pt"
    train(smoke_samples(), path, {"epochs": 1})
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        load_checkpoint(path)


@pytest.mark.parametrize("command", ["predict", "evaluate"])
def test_cli_inference_preserves_checkpoint_method(tmp_path, monkeypatch, command):
    from aurora_visual.cli import main

    path, output = tmp_path / "mean.pt", tmp_path / "output.json"
    train(smoke_samples(), path, {"epochs": 1, "method": "mean-region"})
    args = ["aurora", command, "--smoke", "--checkpoint", str(path), "--output", str(output)]
    monkeypatch.setattr(sys, "argv", args)
    main()
    result = json.loads(output.read_text())
    report = result if command == "predict" else result["evaluation"]
    assert report["method"] == "mean-region"
    if command == "predict":
        model, _, _ = load_checkpoint(path)
        sample = next(s for s in smoke_samples() if s["split"] == "test")
        expected = model(**sample["inputs"], method="mean-region")["probabilities"]
        assert torch.allclose(torch.tensor(report["predictions"][0]["probabilities"]), expected)
    monkeypatch.setattr(sys, "argv", [*args, "--method", "max-region"])
    main()
    result = json.loads(output.read_text())
    assert (result if command == "predict" else result["evaluation"])["method"] == "max-region"


@pytest.mark.parametrize("command", ["predict", "evaluate"])
def test_cli_rejects_changed_features_with_same_dimension(tmp_path, monkeypatch, command):
    import aurora_visual.cli as cli

    path, output = tmp_path / "model.pt", tmp_path / "output.json"
    samples = smoke_samples()
    train(samples, path, {"epochs": 1})
    for sample in samples:
        sample["feature_config"] = {**sample["feature_config"], "preprocessing": "different-grid"}
    monkeypatch.setattr(cli, "smoke_samples", lambda _: samples)
    monkeypatch.setattr(
        sys, "argv", ["aurora", command, "--smoke", "--checkpoint", str(path), "--output", str(output)]
    )
    with pytest.raises(ValueError, match="CHECKPOINT_FEATURE_MISMATCH"):
        cli.main()
    assert not output.exists()


def test_pixel_robustness_rejects_mismatch_except_explicit_fixture_smoke(tmp_path):
    from aurora_visual.alignment.model import VisualHead
    from aurora_visual.evaluation.robustness import evaluate_pixels
    from PIL import Image

    model = VisualHead()
    meta = {"feature_config": smoke_samples()[0]["feature_config"], "data_kind": "fixture"}
    args = (model, Image.new("RGB", (32, 24), "red"), "Bidang ini berwarna merah", "id", tmp_path)
    with pytest.raises(ValueError, match="CHECKPOINT_FEATURE_MISMATCH"):
        evaluate_pixels(*args, checkpoint_metadata=meta)
    result = evaluate_pixels(*args, checkpoint_metadata=meta, allow_fixture_mismatch=True)
    assert len(result["variants"]) == 4
    assert all(v["feature_compatibility"] == "fixture_override_unvalidated" for v in result["variants"])
    with pytest.raises(ValueError, match="CHECKPOINT_FEATURE_MISMATCH"):
        evaluate_pixels(
            *args, checkpoint_metadata={**meta, "data_kind": "research"}, allow_fixture_mismatch=True
        )


def test_split_leakage_rejected(tmp_path):
    sample = dict(
        sample_id="one",
        image="x.png",
        image_sha256="a" * 64,
        caption="caption",
        language="en",
        dataset="local",
        split="train",
        event_id="event",
        source_group="source1",
        parent_id=None,
        license="test",
        source_url=None,
        post_label=None,
        annotations=[],
        data_kind="fixture",
    )
    other = {**sample, "sample_id": "two", "split": "test", "source_group": "source2"}
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(sample) + "\n" + json.dumps(other))
    with pytest.raises(ValueError, match="leakage"):
        load_manifest(path, require_images=False)


def test_metrics_grouping_and_probes():
    rows = [{"group": "one", "truth": 0, "prediction": 0}] * 8
    assert grouped_bootstrap(rows)["low"] is None
    rows += [{"group": "two", "truth": 1, "prediction": 0}]
    assert grouped_bootstrap(rows, repetitions=20)["units"] == 2
    assert iou((0, 0, 1, 1), (0.5, 0.5, 1, 1)) == 0.25
    sample = smoke_samples()[0]
    image_only = modality_probe([sample], "image-only")[0]
    assert image_only["inputs"]["text"].count_nonzero() == 0
    assert sample["inputs"]["text"].count_nonzero() > 0
    assert extraction_metrics([], [])["f1"] == 0
