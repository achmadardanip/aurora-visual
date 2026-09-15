"""Multi-evidence fusion per atom: visual-only, evidence-only, weighted rules.

Rule baselines produce labels/heuristic scores only — probabilities stay null
and raw scores live in extensions (contract invariant 3). Independence groups
cap syndicated copies: one story copied to ten domains is one vote. Forensic
AI-detector signals NEVER enter the truth feature vector.
"""

from dataclasses import dataclass

FACT_LABELS = ("Supported", "Contradicted", "InsufficientEvidence")

STANCE_WEIGHT = {"Supports": 1.0, "Contradicts": -1.0, "NotRelevant": 0.0, "Unclear": 0.0}
CONTENT_WEIGHT = {"full": 1.0, "excerpt_only": 0.7, "snippet_only": 0.4, "unavailable": 0.0}

# Documented hypothesis weights (tuning only on validation data).
RULE_WEIGHTS = {"visual": 0.45, "evidence": 0.55}


@dataclass
class FusionFeatures:
    atom_id: str
    visual_status: str | None = None  # S/C/U visual label if present
    visual_supported: float = 0.0
    visual_contradicted: float = 0.0
    visual_unobservable: float = 0.0
    observability: float | None = None
    support_groups: int = 0
    contradict_groups: int = 0
    unclear_groups: int = 0
    support_weight: float = 0.0
    contradict_weight: float = 0.0
    eligible_count: int = 0
    conflicts: bool = False
    missing_visual: bool = True
    missing_evidence: bool = True


def build_features(atom, evidence, evidence_links, visual_assessments) -> FusionFeatures:
    """Per-atom features from visual assessments and verified eligible evidence."""
    features = FusionFeatures(atom_id=atom.atom_id)
    assessment = next((v for v in visual_assessments if v.atom_id == atom.atom_id), None)
    if assessment:
        features.visual_status = assessment.visual_status
        features.visual_supported = 1.0 if assessment.visual_status == "Supported" else 0.0
        features.visual_contradicted = 1.0 if assessment.visual_status == "Contradicted" else 0.0
        features.visual_unobservable = 1.0 if assessment.visual_status == "Unobservable" else 0.0
        features.observability = assessment.observability_score
        features.missing_visual = False
    links_by_group: dict[str, list] = {}
    evidence_map = {e.evidence_id: e for e in evidence}
    for link in evidence_links:
        if link.atom_id != atom.atom_id or link.stance not in ("Supports", "Contradicts", "Unclear"):
            continue
        item = evidence_map.get(link.evidence_id)
        if item is None:
            continue
        if item.provenance.temporal_eligible is False:
            continue  # future evidence is visible but not eligible
        features.missing_evidence = False
        features.eligible_count += 1
        group = item.provenance.independence_group_id
        links_by_group.setdefault(group, []).append((link, item))
    for group, entries in links_by_group.items():
        supports = [pair for pair, _ in entries if pair.stance == "Supports"]
        contradicts = [pair for pair, _ in entries if pair.stance == "Contradicts"]
        # Conservative intra-group aggregation: strongest consistent stance.
        if supports and contradicts:
            features.conflicts = True
            continue
        for link, item in entries:
            weight = (
                STANCE_WEIGHT[link.stance]
                * CONTENT_WEIGHT.get(item.content.status, 0.0)
                * (0.5 + 0.5 * (item.credibility_score if item.credibility_score is not None else 0.5))
                * (1.0 if item.provenance.temporal_eligible is not False else 0.0)
            )
            if link.stance == "Supports":
                features.support_weight += weight
            elif link.stance == "Contradicts":
                features.contradict_weight += -abs(weight)
        if supports:
            features.support_groups += 1
        if contradicts:
            features.contradict_groups += 1
        if not supports and not contradicts:
            features.unclear_groups += 1
    # Material conflict also spans independent groups with opposing stances.
    if features.support_groups and features.contradict_groups:
        features.conflicts = True
    return features


def visual_only(features: FusionFeatures) -> tuple[str, str]:
    """Baseline 1: visual S/C/U mapped conservatively to factual labels."""
    if features.missing_visual:
        return "InsufficientEvidence", "visual_tidak_tersedia"
    if features.visual_status == "Supported":
        return "Supported", "visual_supported"
    if features.visual_status == "Contradicted":
        return "Contradicted", "visual_contradicted"
    # Unobservable visual is NOT factual InsufficientEvidence on its own,
    # but with no other evidence route it stays unresolved.
    return "InsufficientEvidence", "visual_unobservable"


def evidence_only(features: FusionFeatures) -> tuple[str, str]:
    """Baseline 2: external evidence stance aggregation only."""
    if features.missing_evidence or features.eligible_count == 0:
        return "InsufficientEvidence", "no_eligible_evidence"
    if features.contradict_groups and not features.support_groups:
        return "Contradicted", "evidence_contradicts"
    if features.support_groups and not features.contradict_groups:
        return "Supported", "evidence_supports"
    if features.support_groups and features.contradict_groups:
        return "InsufficientEvidence", "evidence_conflict"
    return "InsufficientEvidence", "evidence_unclear"


def rule_fusion(features: FusionFeatures) -> tuple[str, str, dict]:
    """Baseline 3: weighted rule fusion of visual + evidence scores."""
    visual_score = features.visual_supported - features.visual_contradicted
    evidence_score = features.support_weight - abs(features.contradict_weight)
    evidence_signal = 0.0
    if features.support_groups or features.contradict_groups:
        evidence_signal = max(
            min(evidence_score / max(1, features.support_groups + features.contradict_groups), 1.0), -1.0
        )
    score = RULE_WEIGHTS["visual"] * visual_score + RULE_WEIGHTS["evidence"] * evidence_signal
    diagnostics = {
        "visual_score": round(visual_score, 4),
        "evidence_signal": round(evidence_signal, 4),
        "fused_score": round(score, 4),
        "support_groups": features.support_groups,
        "contradict_groups": features.contradict_groups,
        "conflicts": features.conflicts,
    }
    if features.conflicts:
        return "InsufficientEvidence", "conflicting_evidence", diagnostics
    if score >= 0.5 and (features.support_groups > 0 or features.visual_status == "Supported"):
        return "Supported", "fusion_supports", diagnostics
    if score <= -0.5 and (features.contradict_groups > 0 or features.visual_status == "Contradicted"):
        return "Contradicted", "fusion_contradicts", diagnostics
    return "InsufficientEvidence", "fusion_unresolved", diagnostics


def feature_vector(features: FusionFeatures) -> list[float]:
    """Ordered numeric features for the trainable head (schema is hashed)."""
    return [
        features.visual_supported,
        features.visual_contradicted,
        features.visual_unobservable,
        0.0 if features.missing_visual else 1.0,
        features.observability if features.observability is not None else 0.0,
        min(features.support_groups, 5) / 5.0,
        min(features.contradict_groups, 5) / 5.0,
        min(features.unclear_groups, 5) / 5.0,
        max(-1.0, min(1.0, features.support_weight)),
        max(-1.0, min(1.0, features.contradict_weight)),
        min(features.eligible_count, 10) / 10.0,
        0.0 if features.missing_evidence else 1.0,
        1.0 if features.conflicts else 0.0,
    ]


FEATURE_NAMES = [
    "visual_supported",
    "visual_contradicted",
    "visual_unobservable",
    "visual_present",
    "observability",
    "support_groups",
    "contradict_groups",
    "unclear_groups",
    "support_weight",
    "contradict_weight",
    "eligible_count",
    "evidence_present",
    "conflicts",
]
