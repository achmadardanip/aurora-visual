import argparse
import json
from pathlib import Path

from app.models.contract import AuroraBundle, sha, strict_json

from aurora_visual.training.data import import_public, samples_from_manifest, smoke_samples
from aurora_visual.training.engine import check_feature_compatibility, load_checkpoint, train


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(str(path.resolve()))


def main():
    parser = argparse.ArgumentParser(description="AURORA Visual engineering and research CLI")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "resume", "predict", "evaluate", "cache-features", "experiments"):
        p = sub.add_parser(command)
        p.add_argument("--manifest")
        p.add_argument("--smoke", action="store_true")
        p.add_argument("--checkpoint")
        p.add_argument("--output", default=f"artifacts/reports/{command}.json")
        p.add_argument("--config")
        p.add_argument("--epochs", type=int, default=5)
        p.add_argument("--seed", type=int, default=17)
        p.add_argument("--backbone", default="local-color-v1")
        p.add_argument(
            "--method", help="Override alignment; inference defaults to the saved checkpoint method"
        )
        p.add_argument("--split", default="test")
        p.add_argument("--confirmatory", action="store_true")
    p = sub.add_parser("evaluate-parser")
    p.add_argument("--benchmark", default="data/benchmarks/pilot-candidates.jsonl")
    p.add_argument("--parser", choices=["rules", "llm"], default="rules")
    p.add_argument("--output", default="artifacts/reports/parser-benchmark.json")
    p = sub.add_parser("robustness")
    p.add_argument("--manifest")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--backbone", default="local-color-v1")
    p.add_argument("--output", default="artifacts/reports/robustness.json")
    p = sub.add_parser("export-checkpoint")
    p.add_argument("checkpoint")
    p.add_argument("--output", required=True)
    p = sub.add_parser("validate")
    p.add_argument("bundle")
    p = sub.add_parser("schema")
    p.add_argument("--output", default="contracts/aurora.schema.json")
    p = sub.add_parser("weak-supervision")
    p.add_argument("caption")
    p.add_argument("--output", default="artifacts/reports/weak.json")
    p = sub.add_parser("import-dataset")
    p.add_argument("--source", required=True, choices=["newsclippings", "verite", "cosmos", "mmfakebench"])
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--image-root", required=True)
    p.add_argument("--groups", required=True)
    p.add_argument("--split", required=True)
    p.add_argument("--license-note", required=True)
    p.add_argument("--visualnews")
    args = parser.parse_args()
    if args.command == "evaluate-parser":
        from aurora_visual.evaluation.parser_benchmark import evaluate_parser

        write(args.output, evaluate_parser(args.benchmark, args.parser))
        return
    if args.command == "robustness":
        from PIL import Image, ImageOps

        from aurora_visual.evaluation.robustness import evaluate_pixels
        from aurora_visual.training.data import load_manifest

        model, meta, _ = load_checkpoint(args.checkpoint)
        if args.smoke:
            cases = [
                (
                    "synthetic-pixels",
                    Image.new("RGB", (160, 120), "red"),
                    "Bidang ini berwarna merah pada September 2026",
                    "id",
                )
            ]
        elif args.manifest:
            records = load_manifest(args.manifest)
            cases = [
                (
                    r.sample_id,
                    ImageOps.exif_transpose(Image.open(Path(args.manifest).parent / r.image)).convert("RGB"),
                    r.caption,
                    r.language,
                )
                for r in records
                if r.split == "test"
            ]
        else:
            parser.error("Provide --smoke or --manifest")
        write(
            args.output,
            {
                "data_kind": "fixture" if args.smoke else "manifest",
                "research_evaluated": False,
                "checkpoint": meta,
                "results": [
                    {
                        "sample_id": id_,
                        **evaluate_pixels(
                            model,
                            image,
                            caption,
                            language,
                            Path("artifacts/cache"),
                            args.backbone,
                            meta["config"]["method"],
                            checkpoint_metadata=meta,
                            allow_fixture_mismatch=args.smoke,
                        ),
                    }
                    for id_, image, caption, language in cases
                ],
            },
        )
        return
    if args.command == "schema":
        write(args.output, AuroraBundle.model_json_schema())
        return
    if args.command == "validate":
        b = AuroraBundle.model_validate(strict_json(Path(args.bundle).read_bytes()))
        print("Valid AURORA 1.0.0:", b.case_id)
        return
    if args.command == "weak-supervision":
        from aurora_visual.training.weak import perturb

        write(args.output, perturb(args.caption))
        return
    if args.command == "import-dataset":
        print(
            import_public(
                args.source,
                args.input,
                args.output,
                args.image_root,
                args.split,
                args.license_note,
                args.groups,
                args.visualnews,
            )
        )
        return
    if args.command == "export-checkpoint":
        model, meta, _ = load_checkpoint(args.checkpoint)
        from safetensors.torch import save_file

        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_file(model.state_dict(), path)
        write(
            str(path) + ".json",
            {**meta, "weights_sha256": sha(path.read_bytes()), "optimizer_exported": False},
        )
        return
    if not args.smoke and not args.manifest:
        parser.error("Provide --manifest or explicitly select --smoke")
    samples = (
        smoke_samples(args.seed)
        if args.smoke
        else samples_from_manifest(
            args.manifest,
            "artifacts/cache",
            args.backbone,
            include_unlabeled=args.command == "cache-features",
        )
    )
    if args.command == "cache-features":
        write(
            args.output,
            {
                "samples": len(samples),
                "backbone": args.backbone,
                "data_kind": "fixture" if args.smoke else "manifest",
                "cache_directory": "artifacts/cache",
            },
        )
        return
    config = strict_json(Path(args.config).read_bytes()) if args.config else {}
    config = {
        **config,
        "epochs": args.epochs,
        "seed": args.seed,
        "method": args.method or config.get("method", "uot"),
    }
    if args.command in ("train", "resume"):
        meta, history = train(
            samples,
            args.output,
            config,
            args.backbone,
            resume=args.checkpoint if args.command == "resume" else None,
        )
        print(
            json.dumps(
                {
                    "checkpoint": args.output,
                    "trained_steps": meta["trained_steps"],
                    "data_kind": meta["data_kind"],
                    "last_epoch": history[-1],
                },
                indent=2,
            )
        )
        return
    if args.command == "experiments":
        from aurora_visual.evaluation.runner import evaluate, modality_probe

        results = []
        matrix = [
            {"name": m, "method": m}
            for m in ("global", "max-region", "mean-region", "attention", "balanced-ot", "uot")
        ] + [
            {"name": "cosine-only", "cosine_only": True},
            {"name": "without-unmatched", "use_unmatched": False},
            {"name": "without-hard-positives", "hard_positives": False},
            {"name": "text-only", "probe": "text-only"},
            {"name": "image-only", "probe": "image-only"},
        ]
        for run in matrix:
            rows = modality_probe(samples, run["probe"]) if "probe" in run else samples
            path = Path(args.output).parent / "checkpoints" / f"{run['name']}-{args.seed}.pt"
            meta, history = train(
                rows,
                path,
                {**config, **{k: v for k, v in run.items() if k not in ("name", "probe")}},
                args.backbone,
            )
            model, _, _ = load_checkpoint(path)
            result = evaluate(
                model,
                rows,
                args.split,
                meta["config"]["method"],
                confirmatory=args.confirmatory,
                interventions=False,
            )
            results.append(
                {"name": run["name"], "checkpoint": str(path), "metrics": result, "history": history}
            )
        write(
            args.output,
            {
                "seed": args.seed,
                "experiments": results,
                "optional_parser_comparison": {
                    "status": "unconfigured",
                    "reason": "Run optional parser on reviewed gold manifest after provider setup",
                },
                "note": "Smoke fixtures verify code only; no scientific claims",
            },
        )
        return
    if not args.checkpoint:
        parser.error("--checkpoint required")
    model, meta, _ = load_checkpoint(args.checkpoint)
    check_feature_compatibility(meta, samples)
    method = args.method or meta["config"]["method"]
    if args.confirmatory and meta["data_kind"] != "research":
        raise ValueError("Fixture checkpoint forbidden in confirmatory evaluation")
    if args.split in ("test", "validation", "calibration"):
        trained_groups = meta.get("split_groups", {}).get("train", [])
        for sample in samples:
            if sample["split"] != args.split:
                continue
            for group in trained_groups:
                if any(group[key] == sample[key] for key in ("group", "source_group", "image_sha256")):
                    raise ValueError("Evaluation overlaps checkpoint training group/image/source")
    if args.command == "predict":
        import torch

        with torch.inference_mode():
            predictions = [
                {
                    "id": s["id"],
                    "labels": meta["label_map"],
                    "probabilities": model(**s["inputs"], method=method)["probabilities"].tolist(),
                }
                for s in samples
                if s["split"] == args.split
            ]
        write(
            args.output,
            {"checkpoint": meta, "method": method, "predictions": predictions, "calibration": "unavailable"},
        )
    else:
        from aurora_visual.evaluation.runner import evaluate

        write(
            args.output,
            {
                "checkpoint": meta,
                "evaluation": evaluate(model, samples, args.split, method, confirmatory=args.confirmatory),
            },
        )


if __name__ == "__main__":
    main()
