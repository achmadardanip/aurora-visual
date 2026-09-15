"""Claim-level aggregation over atomic verdicts.

Essential atoms and importance weights are fixed from predicate/role and
check_worthiness BEFORE looking at predicted labels. Conjunctive logic:
one essential atom contradicted by adequate evidence can contradict the
claim; Supported requires ALL essential atoms resolved Supported with no
material conflict; anything else stays InsufficientEvidence. Supported-
weight coverage uses the FULL essential weight as denominator.
"""

from aurora_decision.core.fusion import FusionFeatures

ESSENTIAL_ROLES = {"actor", "action", "time", "location", "quantity"}


def essential_atoms(atoms) -> list[tuple[object, float]]:
    """(atom, weight) pairs; documented rule, computed before label lookup."""
    pairs = []
    for atom in atoms:
        weight = max(atom.check_worthiness, 0.5 if atom.role in ESSENTIAL_ROLES else 0.0)
        if weight > 0:
            pairs.append((atom, min(1.0, weight)))
    if not pairs:  # degenerate: everything below threshold -> treat all as essential
        pairs = [(atom, 1.0) for atom in atoms]
    return pairs


def aggregate_claim(atoms, decisions: dict[str, dict], features: dict[str, FusionFeatures]):
    """Return (base_label, rationale, coverage, details)."""
    essentials = essential_atoms(atoms)
    total_weight = sum(weight for _, weight in essentials) or 1.0
    supported_weight = 0.0
    contradicted = []
    unresolved = []
    for atom, weight in essentials:
        decision = decisions.get(atom.atom_id)
        if decision is None:
            unresolved.append(atom.atom_id)
            continue
        if decision["base_label"] == "Supported" and not decision.get("conflicts"):
            supported_weight += weight
        elif decision["base_label"] == "Contradicted" and decision.get("evidence_backed"):
            contradicted.append(atom.atom_id)
        else:
            unresolved.append(atom.atom_id)
    coverage = supported_weight / total_weight
    if contradicted:
        return (
            "Contradicted",
            f"{len(contradicted)} atom esensial dibantah bukti layak.",
            coverage,
            {"contradicted": contradicted, "unresolved": unresolved},
        )
    if not unresolved and coverage >= 0.999:
        return (
            "Supported",
            "Semua atom esensial terselesaikan Didukung tanpa konflik material.",
            coverage,
            {"contradicted": [], "unresolved": []},
        )
    return (
        "InsufficientEvidence",
        f"{len(unresolved)} atom esensial belum terselesaikan; bobot dukungan {coverage:.0%} dari seluruh bobot esensial.",
        coverage,
        {"contradicted": [], "unresolved": unresolved},
    )
