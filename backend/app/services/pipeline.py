import json
import resource
import time
from datetime import datetime, timezone

import torch
from aurora_visual.alignment.uot import align
from aurora_visual.atomization.parser import HiveVLMAtomizer, RuleAtomizer, StructuredLLMAtomizer
from aurora_visual.entailment.reasoning import assess
from aurora_visual.hive import (
    PROVIDER_REGION_START,
    HiveError,
    HiveV2Client,
    HiveV3Client,
    detector_for_error,
    fuse_observations,
    group_v2_capabilities,
    model_capabilities,
    normalize_observations,
    normalize_translation,
    normalize_v2,
    ocr_from_models,
    regions_for_detections,
    select_models,
    stage1_from_v2,
    v2_capability_status,
    warning_for_error,
)
from aurora_visual.ocr.engine import recognize
from aurora_visual.origin_screen import screen_image
from aurora_visual.vision.features import extract_multi, feature_identity, model_inputs
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
    assets = []
    for media in bundle.input.images:
        source_path, stored, transform = media_service.resolve(media, owner)
        preview_path, _, _ = media_service.resolve(media, owner, preview=True)
        assets.append(
            {
                "ref": stored,
                "source_path": source_path,
                "preview_path": preview_path,
                "image": Image.open(preview_path).convert("RGB"),
                "transform": transform,
                "screening": screen_image(source_path, stored, bundle.mode),
            }
        )
    primary = assets[0]
    use_hive = bundle.mode == "live" and settings.hive_enabled and options.get("provider", "local") == "hive"
    parser_name = options.get("parser", "rules")
    if not use_hive and (parser_name == "hive-vlm" or options.get("translation_shadow", False)):
        raise ValueError("HIVE_PROVIDER_REQUIRED")
    hive_extension = None
    warnings = []
    hive_v2_models = []
    hive_v2_groups = []
    hive_v2_status = []
    if use_hive:
        hive_extension = {
            "mode": "external",
            "egress": {
                "original_media_stage1": True,
                "normalized_preview_stage3": True,
                "caption_stage2_3": True,
            },
            "stage1": None,
            "stage2": None,
            "stage3": None,
            "provider_status": hive_v2_status,
        }
        groups = group_v2_capabilities(settings)
        configured_capabilities = {
            capability for capabilities in groups.values() for capability in capabilities
        }
        missing_capabilities = {
            "origin",
            "ocr",
            "object",
            "scene",
            "people",
            "logo",
            "celebrity",
        } - configured_capabilities
        hive_v2_status.extend(
            {"capability": capability, "status": "unconfigured"}
            for capability in sorted(missing_capabilities)
        )
        if missing_capabilities:
            warnings.append(warning_for_error("unconfigured", "hive_v2_capabilities"))
        for (key, representation), capabilities in groups.items():
            for asset in assets:
                target = asset["source_path"] if representation == "original" else asset["preview_path"]
                media = asset["ref"]
                try:
                    response = HiveV2Client(key, settings.hive_timeout).submit_media(target)
                    if representation == "original":
                        original_size = asset["transform"].get("original_size", [media.width, media.height])
                        width, height = int(original_size[0]), int(original_size[1])
                    else:
                        width, height = media.width, media.height
                    normalized = normalize_v2(response["payload"], width, height)
                    models = select_models(normalized, capabilities)
                    if representation == "preview":
                        hive_v2_models.extend(models)
                        hive_v2_groups.append((asset, models))
                    capability_states = {}
                    for capability in capabilities:
                        status, error_code = v2_capability_status(normalized, capability)
                        capability_states[capability] = status
                        record = {"capability": capability, "status": status}
                        if error_code:
                            record["error_code"] = error_code
                        hive_v2_status.append(record)
                        if status in {"failed", "unsupported"}:
                            warnings.append(warning_for_error(status, f"hive_v2_{capability}"))
                    if "origin" in capabilities:
                        origin_status = capability_states["origin"]
                        if origin_status == "ok":
                            detectors, metadata = stage1_from_v2({"models": models})
                            asset["screening"]["detectors"] = detectors
                            asset["screening"]["provider_metadata"] = metadata
                        else:
                            asset["screening"]["detectors"] = detector_for_error(
                                "provider_failed" if origin_status == "failed" else "unsupported_output"
                            )
                            asset["screening"]["provider_metadata"] = {
                                "provider": "hive-v2",
                                "status": origin_status,
                                "verification": "provider_observation_not_cryptographically_verified",
                                "observations": [],
                            }
                        asset.setdefault(
                            "stage1",
                            {
                                "provider": "hive-v2",
                                "status": origin_status,
                                "representation": "original_bytes",
                                "coordinate_space": "original_upload",
                                "duration_ms": response["duration_ms"],
                                "model_count": len(models),
                                "entries": normalized["entries"],
                            },
                        )
                except HiveError as exc:
                    warnings.append(warning_for_error(exc.code, "hive_stage1_3"))
                    hive_v2_status.extend(
                        {"capability": capability, "status": exc.code} for capability in capabilities
                    )
                    if "origin" in capabilities:
                        asset["screening"]["detectors"] = detector_for_error(exc.code)
                        asset["screening"]["provider_metadata"] = {
                            "provider": "hive-v2",
                            "status": exc.code,
                            "verification": "provider_observation_not_cryptographically_verified",
                            "observations": [],
                        }
                        asset.setdefault(
                            "stage1",
                            {
                                "provider": "hive-v2",
                                "status": exc.code,
                                "representation": "original_bytes",
                            },
                        )
        if "origin" in missing_capabilities:
            for asset in assets:
                asset["screening"]["detectors"] = detector_for_error("unconfigured")
                asset["screening"]["provider_metadata"] = {
                    "provider": "hive-v2",
                    "status": "unconfigured",
                    "verification": "provider_observation_not_cryptographically_verified",
                    "observations": [],
                }
            hive_extension["stage1"] = {
                "provider": "hive-v2",
                "status": "unconfigured",
                "representation": "original_bytes",
            }
    for asset in assets:
        label = asset["screening"]["decision"]["label"]
        if label == "likely_ai_generated":
            warnings.append(
                Warning(
                    code="ORIGIN_SCREEN_GENERATIVE_MARKER",
                    message="Metadata memuat indikator alat generatif. Ini bukan penilaian kebenaran klaim.",
                    component="origin_screen",
                )
            )
        elif label == "inconclusive":
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
        if parser_name == "hive-vlm" and use_hive:
            try:
                parser = HiveVLMAtomizer(HiveV3Client(settings.hive_v3_secret, settings.hive_timeout))
                parsed = parser.parse(bundle.input.claim_text, bundle.input.language)
                if hive_extension is not None:
                    hive_extension["stage2"] = {
                        "provider": "hive-v3-vlm",
                        "status": "ok",
                        "atom_count": len(parsed.atoms),
                    }
            except (HiveError, ValueError) as exc:
                code = exc.code if isinstance(exc, HiveError) else "atomizer_invalid"
                warnings.append(warning_for_error(code, "atomizer"))
                parsed = RuleAtomizer().parse(bundle.input.claim_text, bundle.input.language)
                if hive_extension is not None:
                    hive_extension["stage2"] = {
                        "provider": "hive-v3-vlm",
                        "status": "failed_fallback_rules",
                        "error_code": code,
                    }
        else:
            parser = (
                StructuredLLMAtomizer(settings.llm_url, settings.llm_model, settings.llm_allowed_origins)
                if parser_name == "llm"
                else RuleAtomizer()
            )
            parsed = parser.parse(bundle.input.claim_text, bundle.input.language)
        atoms, parser_config = parsed.atoms, parsed.config
        warnings.extend(parsed.warnings)
    if use_hive and options.get("translation_shadow"):
        key = settings.hive_v2_key("translation")
        caption = bundle.input.claim_text
        if len(caption) > 512:
            warnings.append(warning_for_error("unsupported_length", "hive_translation"))
            hive_extension["translation_shadow"] = {
                "status": "unsupported_length",
                "canonical_caption_unchanged": True,
            }
        elif not key:
            warnings.append(warning_for_error("unconfigured", "hive_translation"))
            hive_extension["translation_shadow"] = {
                "status": "unconfigured",
                "canonical_caption_unchanged": True,
            }
        else:
            try:
                translated = HiveV2Client(key, settings.hive_timeout).translate(
                    caption,
                    bundle.input.language.split("-")[0],
                    "en",
                )
                hive_extension["translation_shadow"] = {
                    "status": "ok",
                    "text": normalize_translation(translated),
                    "canonical_caption_unchanged": True,
                }
            except (HiveError, ValueError) as exc:
                code = exc.code if isinstance(exc, HiveError) else "translation_invalid"
                warnings.append(warning_for_error(code, "hive_translation"))
                hive_extension["translation_shadow"] = {
                    "status": "failed",
                    "canonical_caption_unchanged": True,
                }
    parse_ms = (time.perf_counter() - clock) * 1000
    progress("Menghitung OCR dan representasi region")
    ocr = []
    for asset in assets:
        asset_ocr, ocr_warnings = recognize(asset["image"], settings.ocr_lang)
        ocr.extend(asset_ocr)
        warnings.extend(ocr_warnings)
    if use_hive and hive_v2_groups:
        hive_ocr, hive_ocr_provenance = ocr_from_models(hive_v2_models, bundle.input.language)
        ocr.extend(hive_ocr)
        provider_regions = []
        for index, (asset, asset_models) in enumerate(hive_v2_groups):
            detections = [
                detection
                for model in asset_models
                if model.get("detections")
                for detection in model["detections"]
            ]
            provider_regions.extend(
                regions_for_detections(
                    detections,
                    asset["ref"],
                    run_id,
                    start=PROVIDER_REGION_START + len(assets) * 100_000 + index * 10_000,
                )
            )
        hive_extension["stage3"] = {
            "provider": "hive-v2",
            "representation": "normalized_preview",
            "models": [
                {
                    "model": model["model"],
                    "model_version": model["model_version"],
                    "model_type": model["model_type"],
                    "capabilities": sorted(model_capabilities(model)),
                    "classes": model["classes"][:50],
                    "detections": model["detections"][:50],
                    "people_counts": model["people_counts"][:20],
                    "block_text": model["block_text"][:20],
                    "interpretation": "diagnostic_observation_only",
                }
                for model in hive_v2_models[:20]
            ],
            "ocr_provenance": hive_ocr_provenance,
            "regions": [region.model_dump(mode="json") for region in provider_regions],
        }
    elif use_hive:
        hive_extension["stage3"] = {
            "provider": "hive-v2",
            "representation": "normalized_preview",
            "models": [],
            "status": "unconfigured_or_unsupported",
            "ocr_provenance": [],
            "regions": [],
        }
    features = extract_multi(
        [asset["image"] for asset in assets],
        [asset["ref"] for asset in assets],
        atoms,
        run_id,
        settings.data_dir / "cache",
        backbone,
        top_k,
        bundle.input.language,
        bundle.input.claim_text,
        openclip_model=settings.openclip_model,
        openclip_pretrained=settings.openclip_pretrained,
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
    if use_hive and settings.hive_v3_secret:
        try:
            vlm_client = HiveV3Client(settings.hive_v3_secret, settings.hive_timeout)
            vlm_observations = []
            for index, asset in enumerate(assets):
                raw_observations = vlm_client.observe(
                    asset["preview_path"], "image/png", bundle.input.claim_text, atoms
                )
                asset_observations = normalize_observations(
                    raw_observations, {atom.atom_id for atom in atoms}
                )
                vlm_observations.extend(asset_observations)
                assessments = fuse_observations(
                    assessments,
                    asset_observations,
                    atoms,
                    asset["ref"],
                    run_id,
                    start=PROVIDER_REGION_START + 400_000 + index * 100_000,
                )
            existing = hive_extension.get("stage3") or {}
            hive_extension["stage3"] = {
                **existing,
                "vlm": {
                    "provider": "hive-v3-vlm",
                    "status": "ok",
                    "observations": vlm_observations,
                    "interpretation": "probabilistic_visual_observation_requiring_review",
                },
            }
        except (HiveError, ValueError) as exc:
            code = exc.code if isinstance(exc, HiveError) else "observation_invalid"
            warnings.append(warning_for_error(code, "hive_vlm_observation"))
            existing = hive_extension.get("stage3") or {}
            hive_extension["stage3"] = {
                **existing,
                "vlm": {"provider": "hive-v3-vlm", "status": code},
            }
    elif use_hive:
        warnings.append(warning_for_error("unconfigured", "hive_vlm_observation"))
        existing = hive_extension.get("stage3") or {}
        hive_extension["stage3"] = {
            **existing,
            "vlm": {"provider": "hive-v3-vlm", "status": "unconfigured"},
        }
    logits = None
    checkpoint_meta = None
    if options.get("head", "heuristic") == "trained":
        if bundle.mode != "live":
            raise ValueError("Trained checkpoint may only run in live mode")
        from aurora_visual.training.engine import load_checkpoint

        checkpoint = settings.checkpoint
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
    partial = any(
        w.code in ("OCR_UNAVAILABLE", "OCR_FAILED", "UOT_NONCONVERGENCE") or w.code.startswith("HIVE_")
        for w in warnings
    )
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
        "transform": primary["transform"],
        "screening": primary["screening"],
        "images": [
            {
                "asset_id": asset["ref"].asset_id,
                "transform": asset["transform"],
                "screening": asset["screening"],
                "stage1": asset.get("stage1"),
            }
            for asset in assets
        ],
        "hive": hive_extension,
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
                    "screening_version": primary["screening"]["version"],
                }
            )
        ),
        "artifacts": [
            {"path": f"artifacts/{run_id}/{p.name}", "sha256": sha(p.read_bytes())}
            for p in (artifact, feature_file)
        ],
    }
    return AuroraBundle.model_validate(bundle.model_dump(mode="json"))
