"""Module 2 retrieval pipeline: query -> parallel providers -> evidence bundle.

Runs the deterministic query plan against all enabled providers in parallel
with a global time budget, normalizes hits to Evidence, dedups, reranks, and
runs the AI-detector branch (image + text) as a separate semantic track that
never touches factual evidence scores.
"""

import concurrent.futures
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from aurora_evidence.contract import (
    AuroraBundle,
    Evidence,
    ForensicSignal,
    QueryLog,
    Retrieval,
    RunInfo,
    Warning,
    sha,
)
from aurora_evidence.core.bm25 import load_corpus
from aurora_evidence.core.dedup import cluster_evidence
from aurora_evidence.core.providers.demo import DemoFactCheckProvider, demo_forensic_fixtures
from aurora_evidence.core.providers.fetcher import BoundedFetcher
from aurora_evidence.core.providers.gptzero import GPTZeroTextDetector
from aurora_evidence.core.providers.hive_detector import HiveImageDetector
from aurora_evidence.core.providers.image_index import LocalImageIndexProvider
from aurora_evidence.core.providers.interfaces import Registry
from aurora_evidence.core.providers.local_corpus import LocalCorpusProvider
from aurora_evidence.core.providers.tavily import TavilySearchProvider
from aurora_evidence.core.queries import plan_queries
from aurora_evidence.core.ranking import reciprocal_rank_fusion, rerank


def now():
    return datetime.now(timezone.utc)


def build_registry(settings) -> Registry:
    """Compose providers from settings — swapping providers never touches logic."""
    corpus = load_corpus(Path(settings.corpus_path)) if settings.corpus_path else None
    if settings.mode_profile == "local":
        # Profile "local": corpus + demo only; no external calls at all.
        return Registry(
            fact_check_providers=[LocalCorpusProvider(corpus)] if corpus else [],
            fetcher=None,
            image_provenance=LocalImageIndexProvider(corpus) if corpus else None,
            image_detectors=[],
            text_detectors=[],
        )
    providers = Registry(
        search_providers=[TavilySearchProvider()],
        fact_check_providers=[LocalCorpusProvider(corpus)] if corpus else [],
        fetcher=BoundedFetcher(),
        image_provenance=LocalImageIndexProvider(corpus) if corpus else None,
        image_detectors=[HiveImageDetector()],
        text_detectors=[GPTZeroTextDetector()],
    )
    if settings.tavily_api_key:
        providers.search_providers = [TavilySearchProvider()]
    else:
        providers.search_providers = []
    return providers


def _temporal_eligible(published_at, as_of):
    if as_of is None:
        return True  # assessing with current information
    if published_at is None:
        return None  # unknown date: never assumed eligible
    try:
        published = datetime.fromisoformat(published_at.replace("Z", "+00:00"))
        cutoff = datetime.fromisoformat(as_of.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    return published <= cutoff


def _normalize_hit(hit, run_hex, seq, atom_ids, as_of, retrieved_at) -> dict:
    text = hit.full_text if hit.full_text is not None else hit.snippet
    status = "full" if hit.full_text is not None else "snippet_only"
    excerpt = (hit.snippet or "")[:500]
    if excerpt and excerpt not in text:
        excerpt = text[: min(500, len(text))]
    if not text:
        text, excerpt, status = "", "", "unavailable"
    return {
        "evidence_id": f"ev_{run_hex}_{seq:06d}",
        "atom_ids": atom_ids,
        "source": {
            "provider": hit.publisher or "unknown",
            "kind": hit.kind,
            "url": hit.url,
            "title": hit.title or "(tanpa judul)",
            "publisher": hit.publisher,
            "language": hit.language,
            "published_at": hit.published_at,
            "retrieved_at": retrieved_at,
        },
        "content": {"text": text, "excerpt": excerpt, "sha256": sha(text), "status": status},
        "provenance": {
            "original_url": hit.url,
            "archive_url": None,
            "first_seen_at": None,
            "captured_at": None,
            "date_basis": "published_at" if hit.published_at else None,
            "discovery_method": "local_corpus_bm25" if hit.full_text is not None else "search_snippet",
            "duplicate_cluster_id": "pending",
            "independence_group_id": "pending",
            "independence_group_override": hit.independence_group,
            "temporal_eligible": _temporal_eligible(hit.published_at, as_of),
            "usage_note": None,
            "image_matches": [],
        },
        "relevance_score": None,
        "credibility_score": None,
        "rerank_score": None,
        "score_breakdown": {},
    }


def _forensic_signal(result: dict, run_hex: str, seq: int, target: dict, modality: str) -> dict:
    signal = {
        "signal_id": f"fs_{run_hex}_{seq:06d}",
        "target": target,
        "modality": modality,
        "provider": result.get("provider"),
        "model_version": result.get("model_version"),
        "task": "ai_generation_detection",
        "status": result.get("status", "failed"),
        "raw_score": result.get("raw_score"),
        "raw_scale": result.get("raw_scale"),
        "ai_generated_score": None,
        "raw_label": result.get("raw_label"),
        "assessment": "not_assessed",
        "calibration_status": "provider_claimed" if result.get("status") == "ok" else "unknown",
        "applicable_language": None,
        "limitations": result.get("limitations", []),
        "analyzed_at": now().isoformat(),
        "error_code": result.get("error_code"),
    }
    if result.get("status") == "ok" and result.get("raw_score") is not None and result.get("raw_scale"):
        scale = result["raw_scale"]
        score = (result["raw_score"] - scale["min"]) / (scale["max"] - scale["min"])
        if not scale["higher_means_ai"]:
            score = 1 - score
        signal["ai_generated_score"] = round(min(1.0, max(0.0, score)), 6)
        signal["assessment"] = (
            "likely_ai_generated" if signal["ai_generated_score"] >= 0.5 else "likely_human_or_camera"
        )
    elif result.get("status") == "inconclusive":
        signal["assessment"] = "uncertain"
    return signal


def retrieve(bundle: AuroraBundle, settings, media_service, owner, progress=lambda _: None) -> AuroraBundle:
    """Run the full retrieval pipeline and return the updated bundle."""
    started = now()
    clock = time.perf_counter()
    run_id = str(uuid4())
    run_hex = run_id.replace("-", "")
    registry = build_registry(settings)
    as_of = bundle.input.as_of.isoformat() if bundle.input.as_of else None
    retrieved_at = now().isoformat()
    atoms = bundle.analysis.atomic_claims if bundle.analysis else None
    atom_set = bundle.analysis.atom_set_id if bundle.analysis else None
    atom_ids = [a.atom_id for a in atoms] if atoms else []
    ocr_texts = [o.text for o in bundle.analysis.ocr] if bundle.analysis else []
    progress("Menyusun kueri pencarian")
    queries = plan_queries(bundle.input.claim_text, atoms, ocr_texts)

    warnings: list[Warning] = []
    provider_status: list[dict] = []

    def record(provider, capability, status, message=None):
        provider_status.append(
            {"provider": provider, "capability": capability, "status": status, "message": message}
        )

    def run_provider(name, query_text, atom_ref, fn):
        started_ms = time.perf_counter()
        try:
            hits = fn()
            error = None
        except Exception as exc:  # providers are isolated; retrieval continues
            hits, error = None, exc
        duration_ms = (time.perf_counter() - started_ms) * 1000
        executed.append(
            {
                "provider": name,
                "query": query_text,
                "atom_ids": atom_ref,
                "duration_ms": round(duration_ms, 3),
                "result_count": len(hits or []),
                "ok": error is None,
            }
        )
        return name, hits, error

    executed: list[dict] = []
    progress("Mencari bukti dari sumber yang aktif")
    futures = {}
    deadline = time.monotonic() + settings.budget_seconds
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        if bundle.mode == "demo":
            demo = DemoFactCheckProvider()
            futures[
                pool.submit(
                    run_provider,
                    demo.name,
                    bundle.input.claim_text,
                    [],
                    lambda: demo.search(
                        bundle.input.claim_text, bundle.input.language, as_of, settings, 5000
                    ),
                )
            ] = ("fact_check", demo)
        else:
            for provider in registry.fact_check_providers:
                if provider.status(settings) in ("ok",):
                    futures[
                        pool.submit(
                            run_provider,
                            provider.name,
                            bundle.input.claim_text,
                            [],
                            lambda p=provider: p.search(
                                bundle.input.claim_text,
                                bundle.input.language,
                                as_of,
                                settings,
                                int((deadline - time.monotonic()) * 1000),
                            ),
                        )
                    ] = ("fact_check", provider)
                else:
                    record(provider.name, provider.capability, "unconfigured")
            for provider in registry.search_providers:
                if provider.status(settings) == "ok":
                    for query in queries[: settings.max_search_queries]:
                        futures[
                            pool.submit(
                                run_provider,
                                provider.name,
                                query.query,
                                query.atom_ids,
                                lambda p=provider, q=query: p.search(
                                    q.query, settings, int((deadline - time.monotonic()) * 1000)
                                ),
                            )
                        ] = ("search", provider)
                else:
                    record(provider.name, provider.capability, "unconfigured")

    provider_rankings: dict[str, list[str]] = {}
    items: dict[str, dict] = {}
    seq = 0
    for future in concurrent.futures.as_completed(futures):
        kind, provider = futures[future]
        name, hits, error = future.result()
        if error is not None:
            code = getattr(error, "code", type(error).__name__)
            record(provider.name, provider.capability, "failed", code)
            warnings.append(
                Warning(
                    code="PROVIDER_FAILED",
                    message=f"Provider {provider.name} gagal ({code}); jalur lain tetap berjalan.",
                    component="retrieval",
                )
            )
            continue
        if kind == "fact_check":
            record(provider.name, provider.capability, "ok")
        ranking = []
        for hit in hits or []:
            seq += 1
            item = _normalize_hit(hit, run_hex, seq, [], as_of, retrieved_at)
            items[item["evidence_id"]] = item
            ranking.append(item["evidence_id"])
        provider_rankings.setdefault(name, []).extend(ranking)

    # Query log: one honest entry per executed provider query.
    final_query_log = [
        QueryLog(
            query_id=f"q{index:06d}",
            atom_ids=entry["atom_ids"],
            provider=entry["provider"],
            query=entry["query"],
            duration_ms=entry["duration_ms"],
            result_count=entry["result_count"],
        )
        for index, entry in enumerate(executed, start=1)
    ]

    progress("Mengelompokkan duplikat dan mengurutkan bukti")
    item_list = list(items.values())
    cluster_evidence(item_list)
    rrf = reciprocal_rank_fusion(list(provider_rankings.values()))
    selected = rerank(
        {i["evidence_id"]: i for i in item_list},
        rrf,
        as_of=as_of,
        limit=settings.max_evidence,
    )
    # Candidates associated with atoms for coverage scoring (association, not stance).
    if atom_ids:
        claim_tokens = set()
        for atom in atoms:
            claim_tokens.update(atom.statement.lower().split())
        for item in selected:
            text_tokens = set(item["content"]["text"].lower().split())
            overlapping = [
                aid for atom, aid in zip(atoms, atom_ids) if set(atom.statement.lower().split()) & text_tokens
            ]
            item["atom_ids"] = overlapping[:5]

    progress("Menjalankan deteksi AI (gambar dan teks)")
    signals: list[dict] = []
    signal_seq = 0
    forensic_texts = {}
    if bundle.mode == "demo":
        for result in demo_forensic_fixtures(bundle.input.claim_text, sha):
            signal_seq += 1
            signals.append(
                _forensic_signal(
                    {**result, "provider": "demo-fixtures"},
                    run_hex,
                    signal_seq,
                    {"kind": "claim_text", "asset_id": None, "text_sha256": sha(bundle.input.claim_text)},
                    "text",
                )
            )
    else:
        for image_ref in bundle.input.images[:1]:
            for detector in registry.image_detectors:
                if detector.status(settings) != "ok":
                    record(detector.name, detector.capability, "unconfigured")
                    signal_seq += 1
                    signals.append(
                        _forensic_signal(
                            {
                                "status": "unavailable",
                                "error_code": "unconfigured",
                                "provider": detector.name,
                                "limitations": ["Provider belum dikonfigurasi"],
                            },
                            run_hex,
                            signal_seq,
                            {"kind": "image", "asset_id": image_ref.asset_id, "text_sha256": None},
                            "image",
                        )
                    )
                    continue
                try:
                    path, _, _ = media_service.resolve(image_ref, owner)
                    result = detector.detect(path.read_bytes(), image_ref.media_type, settings)
                    result["provider"] = detector.name
                    record(detector.name, detector.capability, "ok" if result["status"] == "ok" else "failed")
                except Exception as exc:
                    result = {
                        "status": "failed",
                        "error_code": type(exc).__name__,
                        "provider": detector.name,
                        "limitations": [],
                    }
                    record(detector.name, detector.capability, "failed")
                signal_seq += 1
                signals.append(
                    _forensic_signal(
                        result,
                        run_hex,
                        signal_seq,
                        {"kind": "image", "asset_id": image_ref.asset_id, "text_sha256": None},
                        "image",
                    )
                )
        for detector in registry.text_detectors:
            if detector.status(settings) != "ok":
                record(detector.name, detector.capability, "unconfigured")
                signal_seq += 1
                signals.append(
                    _forensic_signal(
                        {
                            "status": "unavailable",
                            "error_code": "unconfigured",
                            "provider": detector.name,
                            "limitations": ["Provider belum dikonfigurasi"],
                        },
                        run_hex,
                        signal_seq,
                        {"kind": "claim_text", "asset_id": None, "text_sha256": sha(bundle.input.claim_text)},
                        "text",
                    )
                )
                continue
            result = detector.detect(bundle.input.claim_text, bundle.input.language, settings)
            result["provider"] = detector.name
            record(
                detector.name,
                detector.capability,
                "ok"
                if result["status"] == "ok"
                else ("unconfigured" if result.get("error_code") == "unconfigured" else "failed"),
            )
            signal_seq += 1
            signals.append(
                _forensic_signal(
                    result,
                    run_hex,
                    signal_seq,
                    {"kind": "claim_text", "asset_id": None, "text_sha256": sha(bundle.input.claim_text)},
                    "text",
                )
            )

    partial = bool(warnings) or any(p["status"] in ("failed", "rate_limited") for p in provider_status)
    retrieval = Retrieval(
        run=RunInfo(
            run_id=run_id,
            mode=bundle.mode,
            started_at=started,
            finished_at=now(),
            status="partial" if partial else "completed",
            versions={"service": "1.0.0", "ranker": "rrf-feature-v1", "corpus": settings.corpus_version},
            warnings=warnings,
        ),
        atom_set_id=atom_set,
        evidence_list=[Evidence.model_validate(item) for item in selected],
        forensic_signals=[ForensicSignal.model_validate(signal) for signal in signals],
        query_log=final_query_log,
        provider_status=provider_status,
    )
    output = bundle.model_copy(deep=True)
    output.retrieval = retrieval
    output.decision = None
    output.extensions.setdefault("aurora_contract", {}).setdefault("run_inputs", {})[run_id] = {
        "snapshot_sha256": sha(output.model_dump_json(exclude={"retrieval", "decision"}).encode()),
        "parent_run_ids": {},
    }
    output.extensions["aurora_evidence"] = {
        "planner": {"queries": len(queries), "max": settings.max_search_queries},
        "budget_seconds": settings.budget_seconds,
        "timing_ms": round((time.perf_counter() - clock) * 1000, 3),
        "forensic_texts": forensic_texts,
    }
    return output.model_validate(output.model_dump(mode="json"))
