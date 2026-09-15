"""Module 3 fuse pipeline: verify -> fuse -> calibrate -> policy -> report.

Input routes (calibration must match): visual_only, evidence_only, fused.
The deterministic verifier produces per-(atom, evidence) stances with exact
quotes; fusion computes per-atom candidates; conformal artifacts (when route
and schema match) produce prediction sets; the abstention policy decides the
operational label; aggregation produces the claim verdict; the report is
generated deterministically from structured facts.
"""

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np

from aurora_decision.contract import (
    AtomicDecision,
    AuroraBundle,
    Calibration,
    Decision,
    DecisionReport,
    EvidenceLink,
    RunInfo,
    Warning,
    sha,
)
from aurora_decision.core.aggregation import aggregate_claim
from aurora_decision.core.calibration import (
    CalibrationArtifact,
    CalibrationStore,
    matches_route,
    prediction_set,
)
from aurora_decision.core.fusion import (
    FACT_LABELS,
    FEATURE_NAMES,
    build_features,
    evidence_only,
    feature_vector,
    rule_fusion,
    visual_only,
)
from aurora_decision.core.policies import apply_policy
from aurora_decision.core.reporting import generate_markdown
from aurora_decision.core.train_head import FusionHead, HeadNotTrained
from aurora_decision.core.verifier import verify_atom_against_evidence


def now():
    return datetime.now(timezone.utc)


def input_route(bundle: AuroraBundle) -> tuple[str, list[str]]:
    """Decide the fusion input route and its warnings."""
    has_visual = bool(bundle.analysis and bundle.analysis.visual_assessments)
    has_evidence = bool(
        bundle.retrieval
        and any(e.provenance.temporal_eligible is not False for e in bundle.retrieval.evidence_list)
    )
    warnings = []
    if bundle.retrieval is None:
        route = "visual_only"
        warnings.append("Jalur visual-saja: tidak ada bukti eksternal pada bundle.")
    elif not has_evidence:
        route = "visual_only"
        warnings.append("Semua bukti tidak layak secara temporal atau kosong; jalur visual-saja.")
    elif not has_visual:
        route = "evidence_only"
        warnings.append("Tidak ada penilaian visual; jalur bukti-eksternal-saja.")
    else:
        route = "fused"
    return route, warnings


def feature_schema_sha(route: str) -> str:
    return sha(json.dumps({"features": FEATURE_NAMES, "route": route}, sort_keys=True))


def eligible_evidence(bundle: AuroraBundle):
    """Evidence usable as factual input: temporally eligible with content."""
    if not bundle.retrieval:
        return []
    return [
        e
        for e in bundle.retrieval.evidence_list
        if e.provenance.temporal_eligible is not False and e.content.status != "unavailable"
    ]


def demo_calibration_bundle(alpha: float, route: str):
    """Deterministic synthetic calibration set for demo mode (labeled demo_only).

    Samples are generated from a fixed labeled grid — this trains nothing and
    only exists so the demo route exercises the real conformal math.
    """
    from aurora_decision.core.calibration import calibrate

    roles = ["action", "time", "location", "attribute", "relation"]
    samples = []
    index = 0
    for role in roles:
        for label_index, label in enumerate(FACT_LABELS):
            for offset in (0.1, 0.3, 0.5, 0.7, 0.9):
                index += 1
                samples.append(
                    {
                        "id": f"demo-{index:04d}",
                        "score": round(offset * 0.6 + label_index * 0.08, 4),
                        "group": (role, label),
                    }
                )
    return calibrate(
        samples,
        alpha=alpha,
        input_route=route,
        feature_schema_sha=feature_schema_sha(route),
        method="split_conformal_mondrian",
        created_at=now().isoformat(),
    )


def fuse(bundle: AuroraBundle, settings, progress=lambda _: None) -> AuroraBundle:
    """Compute the Decision for a validated bundle; returns updated bundle."""
    started = now()
    clock = time.perf_counter()
    if bundle.analysis is None:
        raise ValueError("INPUT_ANALYSIS_REQUIRED")
    analysis = bundle.analysis
    atoms = analysis.atomic_claims
    visual = analysis.visual_assessments
    route, route_warnings = input_route(bundle)
    warnings = [Warning(code="ROUTE_" + route.upper(), message=w, component="fusion") for w in route_warnings]

    progress("Memvalidasi kelayakan bukti")
    evidence = eligible_evidence(bundle)

    progress("Memverifikasi pasangan atom-bukti")
    links: list[EvidenceLink] = []
    for atom in atoms:
        for item in evidence:
            if item.atom_ids and atom.atom_id not in item.atom_ids:
                # Evidence explicitly associated with other atoms only.
                continue
            result = verify_atom_against_evidence(atom, item)
            if result.stance in ("Supports", "Contradicts"):
                quote = result.quote or ""
                if quote and quote not in item.content.text:
                    # Exact-quote guarantee: reject instead of inventing.
                    result.stance = "Unclear"
                    quote = None
                links.append(
                    EvidenceLink(
                        evidence_id=item.evidence_id,
                        atom_id=atom.atom_id,
                        stance=result.stance,
                        quote=quote,
                        rationale=result.rationale,
                    )
                )
            elif result.stance == "Unclear":
                links.append(
                    EvidenceLink(
                        evidence_id=item.evidence_id,
                        atom_id=atom.atom_id,
                        stance="Unclear",
                        quote=result.quote if result.quote and result.quote in item.content.text else None,
                        rationale=result.rationale,
                    )
                )

    progress("Menggabungkan bukti per atom")
    features = {atom.atom_id: build_features(atom, evidence, links, visual) for atom in atoms}

    # Trained head (optional): only used when a matching checkpoint exists.
    head, head_probabilities = None, {}
    head_path = Path(settings.head_checkpoint) if settings.head_checkpoint else None
    if head_path and head_path.is_file():
        try:
            loaded = FusionHead.load(head_path)
            if loaded.trained and loaded.metadata.get("feature_schema_sha") == feature_schema_sha(route):
                head = loaded
            else:
                warnings.append(
                    Warning(
                        code="HEAD_SCHEMA_MISMATCH",
                        message="Checkpoint head tidak cocok untuk rute ini; hanya baseline aturan yang dipakai.",
                        component="fusion",
                    )
                )
        except (ValueError, KeyError, json.JSONDecodeError):
            warnings.append(
                Warning(code="HEAD_LOAD_FAILED", message="Checkpoint head gagal dimuat.", component="fusion")
            )
    if head is not None:
        try:
            matrix = [feature_vector(features[a.atom_id]) for a in atoms]
            for atom, p in zip(atoms, head.predict(np.array(matrix))):
                head_probabilities[atom.atom_id] = p
        except HeadNotTrained:
            head = None

    progress("Menghitung himpunan keyakinan (conformal)")
    store = CalibrationStore(Path(settings.data_dir) / "calibration")
    artifact = None
    mismatch_reason = None
    if bundle.mode == "demo":
        artifact = demo_calibration_bundle(settings.alpha, route)
        calibration_status = "demo_only"
    else:
        requested = getattr(settings, "calibration_id", "") or ""
        candidates = [store.load(requested)] if requested else []
        if not candidates or candidates[0] is None:
            candidates = [
                CalibrationArtifact.from_json(entry)
                for entry in store.list()
                if entry.get("alpha") == settings.alpha
            ]
        for candidate in candidates:
            if candidate is None:
                continue
            ok, reason = matches_route(candidate, route, feature_schema_sha(route), list(FACT_LABELS))
            if ok:
                artifact = candidate
                break
            mismatch_reason = reason
        calibration_status = "calibrated" if artifact else "uncalibrated"
        if artifact is None and mismatch_reason:
            warnings.append(
                Warning(
                    code="CALIBRATION_MISMATCH",
                    message=f"Artefak kalibrasi tidak cocok ({mismatch_reason}); keputusan abstain.",
                    component="calibration",
                )
            )

    progress("Menerapkan kebijakan keputusan")
    decisions: dict[str, dict] = {}
    atomic_verdicts: list[AtomicDecision] = []
    for atom in atoms:
        f = features[atom.atom_id]
        if route == "visual_only":
            base, reason = visual_only(f)
        elif route == "evidence_only":
            base, reason = evidence_only(f)
        else:
            base, reason, diagnostics = rule_fusion(f)
        if head_probabilities and atom.atom_id in head_probabilities:
            probabilities = head_probabilities[atom.atom_id]
            # Trained head overrides the rule candidate label.
            base = max(probabilities, key=probabilities.get)
        else:
            probabilities = None
        confidence_set, conformal_diagnostics = None, None
        if artifact is not None and probabilities:
            confidence_set, conformal_diagnostics = prediction_set(probabilities, atom.role, artifact)
        evidence_backed = (
            f.support_groups > 0
            or f.contradict_groups > 0
            or (route == "visual_only" and f.visual_status in ("Supported", "Contradicted"))
        )
        status, abstain, abstention_reasons = apply_policy(
            base,
            confidence_set,
            {
                "calibrated": artifact is not None and calibration_status != "uncalibrated",
                "calibration_mismatch": artifact is None and bool(mismatch_reason),
                "evidence_backed": evidence_backed,
                "conflicts": f.conflicts,
                "verifier_available": True,
            },
        )
        atom_links = [
            link
            for link in links
            if link.atom_id == atom.atom_id and link.stance in ("Supports", "Contradicts")
        ]
        decisions[atom.atom_id] = {
            "base_label": base,
            "evidence_backed": bool(atom_links),
            "conflicts": f.conflicts,
        }
        calibration_model = Calibration(
            status=calibration_status if artifact is not None else "uncalibrated",
            calibration_id=artifact.calibration_id if artifact else None,
            method=artifact.method if artifact else None,
            alpha=artifact.alpha if artifact else None,
            sample_count=artifact.sample_count if artifact else 0,
            group=route,
            fallback_used=None,
            validity_notes=(
                [
                    "Grup/quantile per kandidat label tersimpan di extensions; fallback global dilaporkan per label."
                ]
                if artifact
                else ["Kalibrasi yang cocok tidak tersedia; keputusan abstain dengan base_label untuk audit."]
            ),
        )
        atomic_verdicts.append(
            AtomicDecision(
                atom_id=atom.atom_id,
                base_label=base,
                status=status,
                probabilities=probabilities,
                confidence_set=confidence_set,
                abstention_flag=abstain,
                abstention_reasons=abstention_reasons,
                evidence_links=atom_links,
                visual_atom_ids=[atom.atom_id] if any(v.atom_id == atom.atom_id for v in visual) else [],
                explanation=f"Rute {route}; alasan fusi: {reason}.",
                calibration=calibration_model,
            )
        )

    progress("Mengagregasi putusan klaim")
    claim_label, claim_rationale, coverage, details = aggregate_claim(atoms, decisions, features)
    # Claim-level prediction sets need claim-level calibration (separate artifact,
    # trained by train-claim-head). Without it: confidence_set=null, abstain.
    claim_confidence = None
    claim_reasons = []
    if bundle.mode != "demo" or True:
        # No claim-level calibration artifact exists yet in any mode.
        claim_reasons.append("UNCALIBRATED")
    if details["unresolved"]:
        claim_reasons.append("ESSENTIAL_ATOM_UNRESOLVED")
    claim_abstain = True
    final = "InsufficientEvidence"

    progress("Menyusun laporan keputusan")
    cited = sorted({link.evidence_id for v in atomic_verdicts for link in v.evidence_links})
    unresolved = [f"Atom {aid} belum terverifikasi bukti yang cukup." for aid in details["unresolved"]]
    report = DecisionReport(
        summary=(
            f"Putusan operasional: {final}. {claim_rationale} "
            f"Rute fusi: {route}; {len(evidence)} bukti layak; {len(links)} tautan atom-bukti."
        ),
        key_findings=[
            f"{FACT_LABELS_ID[v.status]}: {next(a.statement for a in atoms if a.atom_id == v.atom_id)}"
            for v in atomic_verdicts
            if v.status in ("Supported", "Contradicted")
        ][:10],
        unresolved_questions=unresolved,
        evidence_ids=cited,
        forensic_signal_ids=[
            s.signal_id for s in (bundle.retrieval.forensic_signals if bundle.retrieval else [])
        ],
        limitations=[
            "Verifier deterministik hanya menilai relasi eksplisit sempit; Unclear bukan bukti.",
            "Indikasi konten AI tidak memengaruhi putusan faktual.",
            *([w.message for w in warnings]),
        ],
        suggested_next_steps=[
            "Periksa kutipan dan sumber pada tautan bukti sebelum mengambil keputusan akhir.",
            "Lengkapi bukti untuk atom yang belum terjawab bila tersedia.",
        ],
        misinformation_category=None
        if final == "InsufficientEvidence"
        else ("misleading_content" if final == "Contradicted" else "accurate_claim"),
    )
    decision = Decision(
        run=RunInfo(
            run_id=str(uuid4()),
            mode=bundle.mode,
            started_at=started,
            finished_at=now(),
            status="completed" if not warnings else "partial",
            versions={
                "service": "1.0.0",
                "verifier": "deterministic-v1",
                "fusion": "rules-v1" + ("+head" if head else ""),
                "calibration": artifact.method if artifact else "none",
            },
            warnings=warnings,
        ),
        atom_set_id=analysis.atom_set_id,
        atomic_verdicts=atomic_verdicts,
        base_label=claim_label,
        final_verdict=final,
        probabilities=None,  # claim-level head is a separate future artifact
        confidence_set=claim_confidence,
        abstention_flag=claim_abstain,
        abstention_reasons=claim_reasons,
        calibration=Calibration(
            status="uncalibrated",
            calibration_id=None,
            method=None,
            alpha=None,
            sample_count=0,
            group="claim:" + route,
            fallback_used=None,
            validity_notes=[
                "Set keyakinan klaim memerlukan kalibrasi klaim tersendiri (train-claim-head); "
                "kalibrasi atom tidak diwariskan ke klaim."
            ],
        ),
        decision_report=report,
        human_review=bundle.decision.human_review if bundle.decision else None,
    )
    output = bundle.model_copy(deep=True)
    output.decision = decision
    output.extensions.setdefault("aurora_contract", {}).setdefault("run_inputs", {})[decision.run.run_id] = {
        "snapshot_sha256": sha(output.model_dump_json(exclude={"decision"}).encode()),
        "parent_run_ids": {
            "analysis": analysis.run.run_id,
            **({"retrieval": bundle.retrieval.run.run_id} if bundle.retrieval else {}),
        },
    }
    output.extensions["aurora_decision"] = {
        "input_route": route,
        "feature_schema_sha": feature_schema_sha(route),
        "coverage": round(coverage, 4),
        "conformal_diagnostics": {
            v.atom_id: {
                "confidence_set": v.confidence_set,
            }
            for v in atomic_verdicts
        },
        "report_markdown": generate_markdown(output),
        "timing_ms": round((time.perf_counter() - clock) * 1000, 3),
    }
    return output.model_validate(output.model_dump(mode="json"))


FACT_LABELS_ID = {
    "Supported": "Didukung",
    "Contradicted": "Bertentangan",
    "InsufficientEvidence": "Bukti tidak cukup",
}
