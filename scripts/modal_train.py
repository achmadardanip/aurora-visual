"""Modal GPU training/tuning app for the AURORA three-state head.

Local usage (fixture plumbing check, minutes of GPU time):

    uv run modal run scripts/modal_train.py --smoke --epochs 3

Research usage (after licensed manifests exist):

    uv run modal volume create aurora-data
    uv run modal volume put aurora-data manifest.jsonl data/manifest.jsonl
    uv run modal volume put aurora-data images/ images/
    uv run modal run scripts/modal_train.py \
        --manifest /aurora-data/manifest.jsonl --backbone openclip \
        --epochs 20 --gpu A10 --tune --trials 25

The app installs the repository dependencies into the Modal image, writes the
manifest/images into an ephemeral data directory (small inputs are passed
inline; large corpora go through the Modal Volume), runs `train` (or Optuna
`tune`) on the requested GPU, saves checkpoints/summaries to a persistent
Modal Volume, and returns metrics. Fixture runs are labeled `data_kind=fixture`
by the training engine and can never be used live.
"""

import json
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]

app = modal.App("aurora-visual-train")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch>=2.6,<3",
        "numpy>=2,<3",
        "pillow>=11,<13",
        "pydantic>=2.10,<3",
        "rfc8785>=0.1.4,<1",
        "optuna>=3.6,<6",
        "python-dotenv>=1,<2",
        "httpx>=0.28,<1",
        "scipy>=1.15,<2",
    )
    .add_local_dir(ROOT / "backend", "/aurora/backend", copy=True)
    .add_local_dir(ROOT / "configs", "/aurora/configs", copy=True)
    .env({"PYTHONPATH": "/aurora/backend", "AURORA_DATA_DIR": "/tmp/aurora"})
    .run_commands(
        "python -c \"import sys; sys.path.insert(0,'/aurora/backend'); import app.models.contract\""
    )
)

artifact_volume = modal.Volume.from_name("aurora-artifacts", create_if_missing=True)


def _load_samples(manifest_text, image_files, backbone, seed):
    import sys

    sys.path.insert(0, "/aurora/backend")
    from aurora_visual.training.data import samples_from_manifest, smoke_samples

    if manifest_text is None:
        return smoke_samples(seed)
    data_dir = Path("/tmp/aurora/data")
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.jsonl"
    manifest_path.write_text(manifest_text)
    for name, payload in (image_files or {}).items():
        target = data_dir / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    return samples_from_manifest(str(manifest_path), "/tmp/aurora/cache", backbone)


@app.function(image=image, timeout=60 * 60 * 4, volumes={"/aurora/artifacts": artifact_volume}, gpu="T4")
def run(
    manifest_text: str | None,
    image_files: dict[str, bytes] | None,
    config: dict,
    epochs: int,
    seed: int,
    backbone: str,
    tune: bool,
    trials: int,
) -> dict:
    import sys

    sys.path.insert(0, "/aurora/backend")
    import torch
    from aurora_visual.training.tuning import tune

    samples = _load_samples(manifest_text, image_files, backbone, seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path("/aurora/artifacts/run")
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "torch": torch.__version__,
        "samples": len(samples),
        "data_kind": "fixture" if any(s["data_kind"] == "fixture" for s in samples) else "research",
        "research_evaluated": False,
    }
    if tune:
        _, best_config, summary = tune(samples, trials, {**config, "epochs": epochs, "seed": seed}, backbone)
        summary_path = out_dir / "tune-summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, default=str))
        result.update(
            {
                "mode": "tune",
                "best_config": best_config,
                "best_validation_macro_f1": summary["best_value_validation_macro_f1"],
                "trials": summary["trials"],
            }
        )
    else:
        from aurora_visual.training.engine import train

        meta, history = train(
            samples, str(out_dir / "checkpoint.pt"), {**config, "epochs": epochs, "seed": seed}, backbone
        )
        result.update(
            {
                "mode": "train",
                "trained_steps": meta["trained_steps"],
                "last_epoch": history[-1] if history else None,
            }
        )
    artifact_volume.commit()
    return result


@app.local_entrypoint()
def main(
    manifest: str | None = None,
    image_dir: str | None = None,
    config: str | None = None,
    epochs: int = 5,
    seed: int = 17,
    backbone: str = "local-color-v1",
    smoke: bool = False,
    tune: bool = False,
    trials: int = 10,
    gpu: str = "T4",
    output: str = "artifacts/reports/modal-run.json",
):
    if not smoke and not manifest:
        raise SystemExit("Provide --manifest or --smoke")
    manifest_text = None
    image_files = {}
    if manifest:
        manifest_text = Path(manifest).read_text()
        if image_dir:
            for path in Path(image_dir).rglob("*"):
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                    image_files[str(path.relative_to(image_dir))] = path.read_bytes()
        if len(manifest_text.encode()) + sum(len(v) for v in image_files.values()) > 200_000_000:
            raise SystemExit(
                "Inline payload too large: stage the manifest and images on a Modal Volume "
                "(see scripts/modal_train.py docstring) instead of inline transfer."
            )
    parsed_config = json.loads(Path(config).read_text()) if config else {}
    result = run.with_options(gpu=gpu).remote(
        manifest_text=manifest_text,
        image_files=image_files,
        config=parsed_config,
        epochs=epochs,
        seed=seed,
        backbone=backbone,
        tune=tune,
        trials=trials,
    )
    report = Path(output)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result, indent=2, default=str))
    print(str(report.resolve()))
