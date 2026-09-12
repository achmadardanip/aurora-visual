import json
import os
import resource
import time
from datetime import datetime, timezone

import torch
from aurora_visual.alignment.uot import align
from aurora_visual.atomization.parser import RuleAtomizer, StructuredLLMAtomizer
from aurora_visual.entailment.reasoning import assess
from aurora_visual.ocr.engine import recognize
from aurora_visual.origin_screen import screen_image
from aurora_visual.vision.features import extract, feature_identity, model_inputs
from PIL import Image
from torch.nn import functional as F

from app.models.contract import Analysis, AuroraBundle, RunInfo, Warning, atom_set_id, canonical, sha


def now():
    return datetime.now(timezone.utc)


def analyze(bundle, run_id, settings, media_service, owner="local", progress=lambda _: None):
    started = now()
    clock = time.perf_counter()
    input_hash = sha(canonical(bundle.model_dump(mode="json")))
    parents = {
        name: getattr(bundle, name).run.run_id
        for name in ("analysis", "retrieval", "decision")
        if getattr(bundle, name)
    }
    options = bundle.extensions.get("aurora_visual", {}).get("options", {})
    method = options.get("alignment", "uot")
    backbone = options.get("backbone", settings.backbone)
    top_k = options.get("top_k", 16)
    progress("Memeriksa asal media dan mengurai klaim")
    source_path, media, transform = media_service.resolve(bundle.input.image, owner)
    path, _, _ = media_service.resolve(media, owner, preview=True)
    image = Image.open(path).convert("RGB")
    screening = screen_image(source_path, media, bundle.mode)
    warnings = []
    if screening["decision"]["label"] == "likely_ai_generated":
        warnings.append(
            Warning(
                code="ORIGIN_SCREEN_GENERATIVE_MARKER",
                message="Metadata memuat indikator alat generatif. Ini bukan penilaian kebenaran klaim.",
                component="origin_screen",
            )
        )
    elif screening["decision"]["label"] == "inconclusive":
        warnings.append(
            Warning(
                code="ORIGIN_SCREEN_INCONCLUSIVE",
                message="Screening asal media tidak konklusif dan tidak mengubah penilaian visual.",
                component="origin_screen",
            )
        )
    if bundle.analysis:
        atoms = bundle.analysis.atomic_claims
        parser_config = {
            "name": "preserved-explicit-atoms",
            "parent_atom_set_id": bundle.analysis.atom_set_id,
        }
    else:
        parser = StructuredLLMAtomizer() if options.get("parser") == "llm" else RuleAtomizer()
        parsed = parser.parse(bundle.input.claim_text, bundle.input.language)
        atoms, parser_config = parsed.atoms, parsed.config
        warnings.extend(parsed.warnings)
    parse_ms = (time.perf_counter() - clock) * 1000
    progress("Menghitung OCR dan representasi region")
    ocr, ocr_warnings = recognize(image, settings.ocr_lang)
    warnings.extend(ocr_warnings)
    features = extract(
        image,
        media,
        atoms,
        run_id,
        settings.data_dir / "cache",
        backbone,
        top_k,
        bundle.input.language,
        bundle.input.claim_text,
    )
    if backbone == "local-color-v1":
        warnings.append(
            Warning(
                code="LIMITED_COLOR_BASELINE",
                message="Baseline lokal hanya mengukur warna bidang. Tidak mengenali objek, aksi, identitas atau lokasi.",
                component="vision",
            )
        )
    elif not bundle.input.language.startswith("en"):
        warnings.append(
            Warning(
                code="BACKBONE_LANGUAGE_UNVALIDATED",
                message="OpenCLIP standar berorientasi bahasa Inggris; mutu Indonesia belum tervalidasi.",
                component="vision",
            )
        )
    progress("Menghitung alignment dan status visual")
    text = F.normalize(features["text"], dim=-1)
    regional = F.normalize(features["visual"], dim=-1)
    cost = (1 - text @ regional.T).clamp_min(0)
    if method == "global":
        global_cost = (
            1 - F.normalize(features["global_text"], dim=-1) @ F.normalize(features["global"], dim=-1)
        ).clamp_min(0)
        cost = global_cost.expand(len(atoms), len(features["regions"]))
    transport = align(cost, method, tolerance=1e-6, max_iter=500)
    if not transport.diagnostics["converged"]:
        warnings.append(
            Warning(
                code="UOT_NONCONVERGENCE",
                message="Solver mencapai batas iterasi; alignment perlu ditinjau.",
                component="alignment",
            )
        )
    assessments = assess(atoms, features, transport, ocr, fixture=bundle.mode == "demo")
    logits = None
    checkpoint_meta = None
    if options.get("head", "heuristic") == "trained":
        if bundle.mode != "live":
            raise ValueError("Trained checkpoint may only run in live mode")
        from aurora_visual.training.engine import load_checkpoint

        checkpoint = os.getenv("AURORA_CHECKPOINT", "")
        if not checkpoint:
            raise ValueError("CHECKPOINT_UNAVAILABLE")
        model, checkpoint_meta, _ = load_checkpoint(checkpoint, for_live=True)
        if (
            checkpoint_meta["backbone_version"] != backbone
            or model.embedding_dim != features["text"].shape[1]
            or checkpoint_meta["feature_config"] != feature_identity(features["config"])
        ):
            raise ValueError("CHECKPOINT_BACKBONE_MISMATCH")
        with torch.inference_mode():
            output = model(**model_inputs(atoms, features), method=method)
        logits = output["logits"].tolist()
        transport, cost = output["transport"], output["cost"]
        labels = checkpoint_meta["label_map"]
        for i, a in enumerate(assessments):
            a.probabilities = {
                label: float(output["probabilities"][i, idx]) for idx, label in enumerate(labels)
            }
            a.inference_kind = "trained"
            a.unmatched_mass = float(transport.unmatched_mass[i])
            # Explicit-evidence gate: no model score may manufacture a contrary observation.
            candidate = labels[int(output["probabilities"][i].argmax())]
            if candidate != a.visual_status:
                a.visual_status = "Unobservable"
                a.supporting_regions, a.contradicting_regions, a.counter_evidence = [], [], None
                a.rationale = "Head dan bukti eksplisit belum sepakat; hasil ditahan sebagai Unobservable. Distribusi head tersedia untuk audit."
    run_dir = settings.data_dir / "artifacts" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    artifact = run_dir / "transport.json"
    artifact.write_text(
        json.dumps(
            {
                "coupling": transport.coupling.detach().tolist(),
                "cost": cost.detach().tolist(),
                "atom_ids": [a.atom_id for a in atoms],
                "region_ids": [r.region_id for r in features["regions"]],
                "regions": [r.model_dump(mode="json") for r in features["regions"]],
                "diagnostics": transport.diagnostics,
                "logits": logits,
                "label_map": ["Supported", "Contradicted", "Unobservable"],
                "alignment_scores": (text @ regional.T).detach().tolist(),
                "global_token": "context only; not specific grounding",
            },
            allow_nan=False,
        )
    )
    feature_file = run_dir / "features.npz"
    import numpy as np

    np.savez_compressed(
        feature_file,
        text=features["text"].numpy(),
        regions=features["visual"].numpy(),
        global_feature=features["global"].numpy(),
    )
    elapsed = (time.perf_counter() - clock) * 1000
    partial = any(w.code in ("OCR_UNAVAILABLE", "OCR_FAILED", "UOT_NONCONVERGENCE") for w in warnings)
    bundle.analysis = Analysis(
        run=RunInfo(
            run_id=run_id,
            mode=bundle.mode,
            started_at=started,
            finished_at=now(),
            status="partial" if partial else "completed",
            versions={
                "service": "1.0.0",
                "parser": parser_config["name"],
                "backbone": backbone,
                "alignment": method,
                "torch": torch.__version__,
                "head": "checkpoint" if logits is not None else "conservative-v1",
            },
            warnings=warnings,
        ),
        atom_set_id=atom_set_id(bundle, atoms),
        atomic_claims=atoms,
        visual_assessments=assessments,
        ocr=ocr,
    )
    bundle.retrieval, bundle.decision = None, None
    bundle.extensions.setdefault("aurora_contract", {}).setdefault("run_inputs", {})[run_id] = {
        "snapshot_sha256": input_hash,
        "parent_run_ids": parents,
    }
    bundle.extensions["aurora_visual"] = {
        **bundle.extensions.get("aurora_visual", {}),
        "parser": parser_config,
        "transform": transform,
        "screening": screening,
        "method": {
            "backbone": backbone,
            "alignment": method,
            "checkpoint": checkpoint_meta,
            "mode": "fixture" if bundle.mode == "demo" else "trained" if logits is not None else "heuristic",
            "region_method": "grid",
            "calibration": "unavailable",
        },
        "diagnostics": transport.diagnostics,
        "cache": {"key": features["cache_key"], "hit": features["cache_hit"]},
        "timing": {
            "parse_ms": parse_ms,
            "total_ms": elapsed,
            "peak_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "peak_rss_note": "bytes on macOS, KiB on Linux; entire job process",
            "cuda_peak_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None,
        },
        "config_sha256": sha(
            canonical(
                {
                    "options": options,
                    "features": features["config"],
                    "parser": parser_config,
                    "screening_version": screening["version"],
                }
            )
        ),
        "artifacts": [
            {"path": f"artifacts/{run_id}/{p.name}", "sha256": sha(p.read_bytes())}
            for p in (artifact, feature_file)
        ],
    }
    return AuroraBundle.model_validate(bundle.model_dump(mode="json"))
